import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders, HttpErrorResponse } from '@angular/common/http';
import { Observable, interval, Subject, throwError, of, timer } from 'rxjs';
import { catchError, switchMap, takeWhile, map, retry, timeout, retryWhen, delayWhen, tap } from 'rxjs/operators';

export interface ProcessingStatus {
  status: string;
  progress?: number;
  stage?: string;
  answer?: string;
  error?: string;
  message?: string;
  alternative_question?: string;
  alternative_index?: number;
  total_alternatives?: number;
  search_query?: string;
  search_index?: number;
  total_searches?: number;
}

export interface ImageUploadResponse {
  message: string;
  image_path: string;
}

@Injectable({
  providedIn: 'root'
})
export class OllamaService {
  private apiUrl = 'http://localhost:5000';
  private maxRetries = 3;
  private pollingInterval = 500; // ms
  private requestTimeout = 60000; // 60 seconds

  constructor(private http: HttpClient) {}

  uploadFiles(files: FileList, sessionId: string): Observable<ProcessingStatus> {
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
    }
    formData.append('session_id', sessionId);

    // First, initiate the upload
    return this.http.post<any>(`${this.apiUrl}/upload`, formData).pipe(
      timeout(this.requestTimeout),
      retry(2), // Retry upload twice if it fails
      catchError(error => {
        console.error('Error uploading files:', error);
        const errorMessage = this.extractErrorMessage(error);
        
        // Check if it's a connection error
        if (error instanceof HttpErrorResponse && (error.status === 0 || error.status === 502 || error.status === 503 || error.status === 504)) {
          return throwError(() => new Error('Cannot connect to the server. Please make sure the backend is running.'));
        }
        
        return throwError(() => new Error(errorMessage));
      }),
      switchMap(response => {
        // Then poll for status updates
        return this.pollProcessingStatus(sessionId);
      })
    );
  }

  uploadImage(image: File, sessionId: string): Observable<ImageUploadResponse> {
    const formData = new FormData();
    formData.append('image', image);
    formData.append('session_id', sessionId);

    return this.http.post<ImageUploadResponse>(`${this.apiUrl}/upload-image`, formData).pipe(
      timeout(this.requestTimeout),
      retry(2), // Retry upload twice if it fails
      catchError(error => {
        console.error('Error uploading image:', error);
        const errorMessage = this.extractErrorMessage(error);
        
        // Check if it's a connection error
        if (error instanceof HttpErrorResponse && (error.status === 0 || error.status === 502 || error.status === 503 || error.status === 504)) {
          return throwError(() => new Error('Cannot connect to the server. Please make sure the backend is running.'));
        }
        
        return throwError(() => new Error(errorMessage));
      })
    );
  }

  private pollProcessingStatus(sessionId: string): Observable<ProcessingStatus> {
    // Create a subject to emit status updates
    const statusSubject = new Subject<ProcessingStatus>();
    let retryCount = 0;
    
    // Poll every 500ms
    const subscription = interval(this.pollingInterval).subscribe(() => {
      this.http.get<ProcessingStatus>(`${this.apiUrl}/status/${sessionId}`).pipe(
        timeout(10000), // 10 second timeout for status requests
        catchError(error => {
          retryCount++;
          console.warn(`Error polling status (attempt ${retryCount}/${this.maxRetries}):`, error);
          
          // Check if it's a connection error
          if (error instanceof HttpErrorResponse && (error.status === 0 || error.status === 502 || error.status === 503 || error.status === 504)) {
            if (retryCount >= this.maxRetries) {
              statusSubject.error(new Error('Cannot connect to the server. Please make sure the backend is running.'));
              subscription.unsubscribe();
              return throwError(() => error);
            }
          } else if (retryCount >= this.maxRetries) {
            statusSubject.error(new Error('Failed to get processing status after multiple attempts.'));
            subscription.unsubscribe();
            return throwError(() => error);
          }
          
          // Return a dummy status to continue polling
          return of({ 
            status: 'processing' as const, 
            progress: 0, 
            message: 'Waiting for server response...' 
          });
        })
      ).subscribe({
        next: (status) => {
          // Reset retry count on successful response
          retryCount = 0;
          statusSubject.next(status);
          
          // If processing is complete or there's an error, stop polling
          if (status.status === 'complete' || status.status === 'error') {
            subscription.unsubscribe();
            statusSubject.complete();
          }
        },
        error: (err) => {
          // This will only be called if the catchError above doesn't handle it
          console.error('Unhandled error polling status:', err);
          statusSubject.error(err);
          subscription.unsubscribe();
        }
      });
    });
    
    return statusSubject.asObservable();
  }

  askQuestion(question: string, sessionId: string, imagePath?: string): Observable<ProcessingStatus> {
    const subject = new Subject<ProcessingStatus>();
    
    // Prepare request body
    const requestBody: any = { 
      question, 
      session_id: sessionId 
    };
    
    // Add image path if provided
    if (imagePath) {
      requestBody.image_path = imagePath;
    }
    
    // Make a POST request to the streaming endpoint
    fetch(`${this.apiUrl}/ask`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    }).then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      // Set up a reader for the stream
      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      
      // Function to read the stream
      const readStream = () => {
        reader.read().then(({ done, value }) => {
          if (done) {
            subject.complete();
            return;
          }
          
          // Decode the chunk and split by newlines
          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split('\n').filter(line => line.trim());
          
          // Process each line as a JSON object
          lines.forEach(line => {
            try {
              const data = JSON.parse(line);
              subject.next(data);
              
              // If there's an error in the response, emit it
              if (data.status === 'error') {
                subject.error(new Error(data.message || 'Unknown error in stream response'));
              }
            } catch (e) {
              console.error('Error parsing JSON:', e, line);
            }
          });
          
          // Continue reading
          readStream();
        }).catch(error => {
          console.error('Error reading stream:', error);
          subject.error(error);
        });
      };
      
      // Start reading the stream
      readStream();
    }).catch(error => {
      console.error('Error fetching from ask endpoint:', error);
      
      // Check if it's likely a connection error
      if (error.message && (
          error.message.includes('Failed to fetch') || 
          error.message.includes('NetworkError') || 
          error.message.includes('Network request failed'))) {
        subject.error(new Error('Cannot connect to the server. Please make sure the backend is running.'));
      } else {
        subject.error(error);
      }
    });
    
    return subject.asObservable();
  }
  
  // Helper method to extract error messages from HTTP responses
  private extractErrorMessage(error: any): string {
    if (error instanceof HttpErrorResponse) {
      // Connection error
      if (error.status === 0) {
        return 'Cannot connect to the server. Please make sure the backend is running.';
      }
      
      if (error.error && error.error.error) {
        return error.error.error;
      } else if (error.statusText) {
        return `${error.status} ${error.statusText}`;
      }
    }
    
    return error.message || 'Unknown error occurred';
  }
}

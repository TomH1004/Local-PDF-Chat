import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class OllamaService {
  private apiUrl = 'http://localhost:5000';

  constructor(private http: HttpClient) {}

  uploadFiles(files: FileList, sessionId: string): Observable<any> {
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
    }
    formData.append('session_id', sessionId);

    return this.http.post(`${this.apiUrl}/upload`, formData);
  }

  askQuestion(question: string, sessionId: string): Observable<any> {
    return this.http.post(`${this.apiUrl}/ask`, { question, session_id: sessionId });
  }
}

import { Component, ViewEncapsulation } from '@angular/core';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';
import { OllamaService, ProcessingStatus, ImageUploadResponse } from './ollama.service';

interface ChatMessage {
  sender: string;
  text: string; // Will store HTML or plain text
}

interface AlternativeQuestion {
  text: string;
  index: number;
}

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css'],
  encapsulation: ViewEncapsulation.None // Allow component styles to affect rendered HTML
})
export class AppComponent {
  title = 'Local PDF Chat';
  pdfSrc: SafeResourceUrl | null = null;
  isProcessing = false;
  isUploading = false;
  isAnswering = false;
  processingProgress = 0;
  answeringProgress = 0;
  currentQuestion = '';
  questionText = '';
  chatMessages: ChatMessage[] = [];
  uploadedFiles: FileList | null = null;
  sessionId = this.generateSessionId();
  processingStage = '';
  alternativeQuestions: AlternativeQuestion[] = [];
  totalAlternatives = 0;
  currentSearchQuery = '';
  searchIndex = 0;
  totalSearches = 0;
  currentImagePath: string | null = null;
  isUploadingImage = false;

  constructor(
    private ollamaService: OllamaService,
    private sanitizer: DomSanitizer
  ) {}

  private generateSessionId(): string {
    return Math.random().toString(36).substr(2, 9);
  }

  onFileSelected(event: Event) {
    // Reset chat and state on new file
    this.resetSession();
    const input = event.target as HTMLInputElement;
    if (!input.files || input.files.length === 0) return;

    this.uploadedFiles = input.files;
    // Display PDF
    this.pdfSrc = this.sanitizer.bypassSecurityTrustResourceUrl(
      URL.createObjectURL(this.uploadedFiles[0])
    );
    // Immediately process the document
    this.analyzeDocuments();
  }

  resetSession() {
    this.pdfSrc = null;
    this.isProcessing = true;
    this.isUploading = false;
    this.isAnswering = false;
    this.processingProgress = 0;
    this.answeringProgress = 0;
    this.processingStage = '';
    this.currentQuestion = '';
    this.questionText = '';
    this.chatMessages = [];
    this.uploadedFiles = null;
    this.sessionId = this.generateSessionId();
    this.alternativeQuestions = [];
    this.totalAlternatives = 0;
    this.currentSearchQuery = '';
    this.searchIndex = 0;
    this.totalSearches = 0;
    this.currentImagePath = null;
  }

  analyzeDocuments() {
    if (!this.uploadedFiles) {
      alert('Please upload a PDF file first.');
      return;
    }
    
    // Validate file type
    const file = this.uploadedFiles[0];
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      this.chatMessages.push({
        sender: 'System',
        text: `<p class="error-message">Invalid file type. Only PDF files are supported. Got: ${file.name}</p>`
      });
      return;
    }
    
    this.isProcessing = true;
    this.isUploading = true;
    this.processingProgress = 0;
    this.processingStage = 'Preparing to upload...';
    
    this.ollamaService.uploadFiles(this.uploadedFiles, this.sessionId)
      .subscribe({
        next: (status: ProcessingStatus) => {
          // Update progress based on status
          this.processingProgress = status.progress;
          
          if (status.message) {
            this.processingStage = status.message;
          }
          
          // Handle completion
          if (status.status === 'complete') {
            this.isProcessing = false;
            this.isUploading = false;
            this.chatMessages.push({
              sender: 'System',
              text: '<p>Document processed successfully. You can now ask questions about the content.</p>'
            });
          }
          
          // Handle error
          if (status.status === 'error') {
            console.error('Processing error:', status.message);
            this.isProcessing = false;
            this.isUploading = false;
            
            // Format the error message for better display
            let errorMessage = status.message || 'Unknown error';
            
            // If it's a nested error message, clean it up
            if (errorMessage.includes('Error processing document:')) {
              errorMessage = errorMessage.replace(/Error processing document:\s+/g, '');
            }
            
            this.chatMessages.push({
              sender: 'System',
              text: `<p class="error-message">Error processing document: ${this.escapeHtml(errorMessage)}</p>`
            });
          }
        },
        error: (err) => {
          console.error('Service error:', err);
          this.isProcessing = false;
          this.isUploading = false;
          
          // Extract error message from the response if available
          let errorMessage = 'Please try again.';
          if (err.error && err.error.error) {
            errorMessage = err.error.error;
          } else if (err.message) {
            errorMessage = err.message;
          }
          
          // Check if it's a connection error
          if (errorMessage.includes('Cannot connect to the server') || 
              errorMessage.includes('Failed to fetch') ||
              errorMessage.includes('NetworkError')) {
            this.chatMessages.push({
              sender: 'System',
              text: `<p class="error-message">
                <strong>Connection Error:</strong> Cannot connect to the backend server. 
                <br><br>
                Please make sure the backend server is running at http://localhost:5000.
                <br><br>
                <em>Tip: Run the backend server with the command: <code>python backend.py</code></em>
              </p>`
            });
          } else {
            this.chatMessages.push({
              sender: 'System',
              text: `<p class="error-message">Error processing document: ${this.escapeHtml(errorMessage)}</p>`
            });
          }
        }
      });
  }

  onImageSelected(event: Event) {
    const input = event.target as HTMLInputElement;
    if (!input.files || input.files.length === 0) return;
    
    const imageFile = input.files[0];
    
    // Validate file type
    const validImageTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
    if (!validImageTypes.includes(imageFile.type)) {
      this.chatMessages.push({
        sender: 'System',
        text: `<p class="error-message">Invalid image type. Supported formats: JPG, PNG, GIF, WEBP</p>`
      });
      return;
    }
    
    // Upload the image
    this.isUploadingImage = true;
    this.ollamaService.uploadImage(imageFile, this.sessionId).subscribe({
      next: (response: ImageUploadResponse) => {
        this.isUploadingImage = false;
        this.currentImagePath = response.image_path;
        
        // Add a message to show the image was uploaded
        this.chatMessages.push({
          sender: 'System',
          text: `<p>Image uploaded successfully. You can now ask questions about it.</p>
                 <div class="uploaded-image-preview">
                   <img src="http://localhost:5000/${response.image_path}" alt="Uploaded image" />
                 </div>`
        });
      },
      error: (err) => {
        this.isUploadingImage = false;
        console.error('Error uploading image:', err);
        
        // Extract error message
        let errorMessage = 'Please try again.';
        if (err.message) {
          errorMessage = err.message;
        }
        
        this.chatMessages.push({
          sender: 'System',
          text: `<p class="error-message">Error uploading image: ${this.escapeHtml(errorMessage)}</p>`
        });
      }
    });
  }

  askQuestion() {
    if (!this.questionText.trim() || this.isAnswering) return;

    // Add user message
    this.chatMessages.push({ 
      sender: 'User', 
      text: `<p>${this.escapeHtml(this.questionText)}</p>` 
    });
    
    const userQuestion = this.questionText;
    this.questionText = '';
    this.isAnswering = true;
    this.answeringProgress = 0;
    this.currentQuestion = userQuestion;
    
    // Reset alternative questions
    this.alternativeQuestions = [];
    this.totalAlternatives = 0;
    this.currentSearchQuery = '';
    this.searchIndex = 0;
    this.totalSearches = 0;

    this.ollamaService.askQuestion(userQuestion, this.sessionId, this.currentImagePath).subscribe({
      next: (status: ProcessingStatus) => {
        // Update progress based on status
        if (status.progress) {
          this.answeringProgress = status.progress;
        }
        
        // Update stage if available
        if (status.stage) {
          this.processingStage = this.formatStage(status.stage);
        }
        
        // Handle alternative questions
        if (status.alternative_question) {
          const altQuestion: AlternativeQuestion = {
            text: status.alternative_question,
            index: status.alternative_index || this.alternativeQuestions.length + 1
          };
          
          // Update or add the alternative question
          const existingIndex = this.alternativeQuestions.findIndex(q => q.index === altQuestion.index);
          if (existingIndex >= 0) {
            this.alternativeQuestions[existingIndex] = altQuestion;
          } else {
            this.alternativeQuestions.push(altQuestion);
          }
          
          // Update total alternatives
          if (status.total_alternatives) {
            this.totalAlternatives = status.total_alternatives;
          }
        }
        
        // Handle search queries
        if (status.search_query) {
          this.currentSearchQuery = status.search_query;
          if (status.search_index) {
            this.searchIndex = status.search_index;
          }
          if (status.total_searches) {
            this.totalSearches = status.total_searches;
          }
        }
        
        // Handle completion
        if (status.status === 'complete' && status.answer) {
          this.isAnswering = false;
          this.currentQuestion = '';
          this.alternativeQuestions = [];
          this.currentSearchQuery = '';
          
          // Process the answer to ensure it's valid HTML
          let answerHtml = '';
          
          if (typeof status.answer === 'string') {
            // Check if the response is already HTML
            if (this.isValidHtml(status.answer)) {
              answerHtml = status.answer;
            } else {
              // Wrap plain text in paragraph tags
              answerHtml = `<p>${this.escapeHtml(status.answer)}</p>`;
            }
          } else {
            // Handle non-string responses
            answerHtml = `<p>${this.escapeHtml(JSON.stringify(status.answer, null, 2))}</p>`;
          }
          
          // Enhance table styling if present
          if (answerHtml.includes('<table>')) {
            answerHtml = this.enhanceTableStyling(answerHtml);
          }
          
          this.chatMessages.push({ 
            sender: 'Assistant', 
            text: answerHtml 
          });
          
          // Clear the current image path after using it for a question
          this.currentImagePath = null;
        }
      },
      error: (err) => {
        console.error(err);
        this.isAnswering = false;
        this.currentQuestion = '';
        this.alternativeQuestions = [];
        this.currentSearchQuery = '';
        
        // Extract error message
        let errorMessage = 'Please try again.';
        if (err.message) {
          errorMessage = err.message;
        }
        
        // Check if it's a connection error
        if (errorMessage.includes('Cannot connect to the server') || 
            errorMessage.includes('Failed to fetch') ||
            errorMessage.includes('NetworkError')) {
          this.chatMessages.push({
            sender: 'System',
            text: `<p class="error-message">
              <strong>Connection Error:</strong> Cannot connect to the backend server. 
              <br><br>
              Please make sure the backend server is running at http://localhost:5000.
              <br><br>
              <em>Tip: Run the backend server with the command: <code>python backend.py</code></em>
            </p>`
          });
        } else {
          this.chatMessages.push({
            sender: 'System',
            text: `<p class="error-message">Error retrieving answer: ${this.escapeHtml(errorMessage)}</p>`
          });
        }
      }
    });
  }
  
  // Format the stage for display
  private formatStage(stage: string): string {
    switch (stage) {
      case 'retrieving': return 'Retrieving relevant information...';
      case 'rephrasing': return 'Analyzing question...';
      case 'searching': return 'Searching for answers...';
      case 'generating': return 'Generating response...';
      default: return `${stage.charAt(0).toUpperCase() + stage.slice(1)}...`;
    }
  }
  
  // Helper method to check if a string is valid HTML
  private isValidHtml(str: string): boolean {
    // Simple check for HTML tags
    return /<\/?[a-z][\s\S]*>/i.test(str);
  }
  
  // Helper method to escape HTML special characters
  private escapeHtml(text: string): string {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
  
  // Helper method to enhance table styling
  private enhanceTableStyling(html: string): string {
    // Add responsive table wrapper
    if (html.includes('<table>')) {
      html = html.replace(/<table>/g, '<div class="table-responsive"><table class="styled-table">');
      html = html.replace(/<\/table>/g, '</table></div>');
      
      // Add header class to th elements
      html = html.replace(/<th>/g, '<th class="table-header">');
      
      // Add zebra striping to rows - using a more compatible approach
      try {
        // Try using matchAll if available (ES2020+)
        const tableRegex = /<table[^>]*>([\s\S]*?)<\/table>/g;
        const matches = html.matchAll(tableRegex);
        
        for (const match of matches) {
          if (match[1]) {
            const tableContent = match[1];
            const rows = tableContent.split('<tr>');
            let newRows = rows[0]; // Keep the first part (before any tr)
            
            for (let i = 1; i < rows.length; i++) {
              const rowClass = i % 2 === 0 ? 'even-row' : 'odd-row';
              newRows += `<tr class="${rowClass}">` + rows[i];
            }
            
            html = html.replace(tableContent, newRows);
          }
        }
      } catch (e) {
        // Fallback method for older browsers/environments
        console.log('Using fallback method for table styling');
        
        // Simple regex-based approach without matchAll
        const tableRegex = /<table[^>]*>([\s\S]*?)<\/table>/;
        const tableMatch = html.match(tableRegex);
        
        if (tableMatch && tableMatch[1]) {
          const tableContent = tableMatch[1];
          const rows = tableContent.split('<tr>');
          let newRows = rows[0]; // Keep the first part (before any tr)
          
          for (let i = 1; i < rows.length; i++) {
            const rowClass = i % 2 === 0 ? 'even-row' : 'odd-row';
            newRows += `<tr class="${rowClass}">` + rows[i];
          }
          
          html = html.replace(tableContent, newRows);
        }
      }
    }
    
    return html;
  }
  
  // Clear the current image
  clearCurrentImage() {
    this.currentImagePath = null;
    this.chatMessages.push({
      sender: 'System',
      text: '<p>Image cleared. Your next question will not reference any image.</p>'
    });
  }
}

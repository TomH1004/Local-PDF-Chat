import { Component } from '@angular/core';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';
import { OllamaService } from './ollama.service';

interface ChatMessage {
  sender: string;
  text: string; // Will store HTML or plain text
}

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'Local PDF Chat';
  pdfSrc: SafeResourceUrl | null = null;
  isProcessing = false;
  questionText = '';
  chatMessages: ChatMessage[] = [];
  uploadedFiles: FileList | null = null;
  sessionId = this.generateSessionId();

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
    this.isProcessing = false;
    this.questionText = '';
    this.chatMessages = [];
    this.uploadedFiles = null;
    this.sessionId = this.generateSessionId();
  }

  analyzeDocuments() {
    if (!this.uploadedFiles) {
      alert('Please upload a PDF file first.');
      return;
    }
    this.isProcessing = true;
    this.ollamaService.uploadFiles(this.uploadedFiles, this.sessionId)
      .subscribe({
        next: () => {
          // Document processed
          this.isProcessing = false;
          this.chatMessages.push({
            sender: 'System',
            text: 'Document processed. You can now ask questions.'
          });
        },
        error: (err) => {
          console.error(err);
          this.isProcessing = false;
          this.chatMessages.push({
            sender: 'System',
            text: 'Error processing document.'
          });
        }
      });
  }

  askQuestion() {
    if (!this.questionText.trim()) return;

    this.chatMessages.push({ sender: 'User', text: this.questionText });
    const userQuestion = this.questionText;
    this.questionText = '';
    this.isProcessing = true;

    this.ollamaService.askQuestion(userQuestion, this.sessionId).subscribe({
      next: (response: any) => {
        this.isProcessing = false;
        // If the backend returns HTML, store it directly
        // If it returns Markdown, parse it or store as is
        const answerHtml = typeof response.answer === 'string'
          ? response.answer
          : JSON.stringify(response.answer, null, 2);
        this.chatMessages.push({ sender: 'Assistant', text: answerHtml });
      },
      error: (err) => {
        console.error(err);
        this.isProcessing = false;
        this.chatMessages.push({
          sender: 'System',
          text: 'Error retrieving answer.'
        });
      }
    });
  }
}

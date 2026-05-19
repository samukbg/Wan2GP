/**
 * Hyperframes API Client for Wan2GP
 * 
 * This example demonstrates how to interact with the Wan2GP Hyperframes endpoints
 * using TypeScript and standard fetch.
 */

export interface HyperframesRenderRequest {
  html?: string;
  html_url?: string;
  files?: Record<string, string>; // filename -> content/url
  fps?: number;
  quality?: 'standard' | 'high';
  format?: 'mp4' | 'webm' | 'mov';
  execution_id?: string;
}

export interface HyperframesTTSRequest {
  text: string;
  voice?: string; // e.g., 'af_sky', 'af_heart'
  speed?: number;
  execution_id?: string;
}

export interface HyperframesStatusResponse {
  status: 'queued' | 'processing' | 'completed' | 'failed';
  progress: number;
  output_path?: string;
  output_url?: string;
  error?: string;
  result?: any; // For transcription
}

export class Wan2GPHyperframes {
  private baseUrl: string;

  constructor(baseUrl: string = 'https://video-est-pc2.samuelbezerra.fr') {
    this.baseUrl = baseUrl.replace(/\/$/, '');
  }

  /**
   * Submit a render job
   */
  async render(payload: HyperframesRenderRequest): Promise<{ execution_id: string }> {
    const response = await fetch(`${this.baseUrl}/hyperframes/render`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return response.json();
  }

  /**
   * Generate speech using Kokoro
   */
  async tts(payload: HyperframesTTSRequest): Promise<{ execution_id: string }> {
    const response = await fetch(`${this.baseUrl}/hyperframes/tts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return response.json();
  }

  /**
   * Get job status and results
   */
  async getStatus(execution_id: string): Promise<HyperframesStatusResponse> {
    const response = await fetch(`${this.baseUrl}/render_status/${execution_id}`);
    return response.json();
  }

  /**
   * Helper to wait for a job to finish
   */
  async waitForCompletion(execution_id: string, intervalMs = 3000): Promise<HyperframesStatusResponse> {
    while (true) {
      const status = await this.getStatus(execution_id);
      if (status.status === 'completed' || status.status === 'failed') {
        return status;
      }
      await new Promise(resolve => setTimeout(resolve, intervalMs));
    }
  }
}

// --- Usage Example ---
/*
const client = new Wan2GPHyperframes();

const run = async () => {
  // 1. Render Video
  const { execution_id } = await client.render({
    html: `<div style="background: navy; color: white; padding: 50px;"><h1>Remote TS Test</h1></div>`,
    files: {
        "style.css": "h1 { font-family: sans-serif; }"
    }
  });
  
  const result = await client.waitForCompletion(execution_id);
  console.log("Render Result:", result);
};
*/

import { ProblemData, PredictionResult } from "../types";

const ML_BACKEND_URL = "http://localhost:5001";

export interface MLPredictionResponse {
  difficulty_class: string;
  difficulty_score: number;
  confidence?: number;
  processing_time: number;
  success: boolean;
  error?: string;
}

export interface MLModelInfo {
  training_samples: number;
  test_samples: number;
  features: number;
  test_accuracy: number;
  test_r2: number;
  test_mae: number;
  test_rmse: number;
  class_distribution: {
    Easy: number;
    Medium: number;
    Hard: number;
  };
}

export async function predictDifficulty(data: ProblemData): Promise<PredictionResult> {
  try {
    const response = await fetch(`${ML_BACKEND_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        title: data.title,
        description: data.description,
        input_description: data.inputDescription,
        output_description: data.outputDescription,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
    }

    const result: MLPredictionResponse = await response.json();

    if (!result.success) {
      throw new Error(result.error || "Prediction failed");
    }

    // Convert ML backend response to frontend format
    return {
      problemClass: result.difficulty_class,
      problemScore: result.difficulty_score,
      reasoning: `Predicted using trained Random Forest models with ${result.confidence ? `${(result.confidence * 100).toFixed(1)}% confidence` : 'high accuracy'}. Processing time: ${result.processing_time}s`,
      keywords: generateKeywords(data),
      complexityAnalysis: {
        time: estimateTimeComplexity(result.difficulty_class),
        space: estimateSpaceComplexity(result.difficulty_class),
        implementationEffort: mapScoreToEffort(result.difficulty_score),
        algorithmicDepth: mapScoreToDepth(result.difficulty_score),
      },
      verdict: generateVerdict(result.difficulty_class, result.difficulty_score),
      pitfalls: generatePitfalls(result.difficulty_class),
      suggestions: generateSuggestions(result.difficulty_class),
    };
  } catch (error) {
    console.error("ML Backend Error:", error);
    
    // Check if backend is running
    if (error instanceof TypeError && error.message.includes('fetch')) {
      throw new Error("ML Backend is not running. Please start the Flask server with: python backend/app.py (runs on port 5001)");
    }
    
    throw new Error(`ML prediction failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

export async function getModelInfo(): Promise<MLModelInfo> {
  try {
    const response = await fetch(`${ML_BACKEND_URL}/model-info`);
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const result = await response.json();
    return result.model_info;
  } catch (error) {
    console.error("Failed to get model info:", error);
    throw error;
  }
}

export async function checkBackendHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${ML_BACKEND_URL}/health`);
    const result = await response.json();
    return result.status === "healthy" && result.models_loaded;
  } catch (error) {
    return false;
  }
}

// Helper functions to generate additional context
function generateKeywords(data: ProblemData): string[] {
  const text = `${data.title} ${data.description} ${data.inputDescription} ${data.outputDescription}`.toLowerCase();
  const keywords: string[] = [];
  
  // Algorithm keywords
  if (text.includes('sort') || text.includes('order')) keywords.push('sorting');
  if (text.includes('graph') || text.includes('node') || text.includes('edge')) keywords.push('graph');
  if (text.includes('tree') || text.includes('binary')) keywords.push('tree');
  if (text.includes('dynamic') || text.includes('dp') || text.includes('memo')) keywords.push('dynamic-programming');
  if (text.includes('greedy')) keywords.push('greedy');
  if (text.includes('array') || text.includes('list')) keywords.push('array');
  if (text.includes('string') || text.includes('text')) keywords.push('string');
  if (text.includes('math') || text.includes('number') || text.includes('calculate')) keywords.push('mathematics');
  if (text.includes('search') || text.includes('find')) keywords.push('search');
  if (text.includes('recursive') || text.includes('recursion')) keywords.push('recursion');
  
  return keywords.length > 0 ? keywords : ['general'];
}

function estimateTimeComplexity(difficulty: string): string {
  switch (difficulty) {
    case 'Easy': return 'O(n) or O(n log n)';
    case 'Medium': return 'O(n²) or O(n log n)';
    case 'Hard': return 'O(n³) or exponential';
    default: return 'O(n)';
  }
}

function estimateSpaceComplexity(difficulty: string): string {
  switch (difficulty) {
    case 'Easy': return 'O(1) or O(n)';
    case 'Medium': return 'O(n) or O(n²)';
    case 'Hard': return 'O(n²) or higher';
    default: return 'O(n)';
  }
}

function mapScoreToEffort(score: number): number {
  return Math.min(10, Math.max(1, Math.round(score)));
}

function mapScoreToDepth(score: number): number {
  return Math.min(10, Math.max(1, Math.round(score)));
}

function generateVerdict(difficulty: string, score: number): string {
  const verdicts = {
    Easy: [
      "A gentle warm-up for the coding muscles.",
      "Perfect for a coffee break challenge.",
      "Even beginners can tackle this one."
    ],
    Medium: [
      "Now we're getting somewhere interesting.",
      "A solid test of algorithmic thinking.",
      "This will separate the wheat from the chaff."
    ],
    Hard: [
      "Buckle up, this is where legends are made.",
      "May the algorithms be ever in your favor.",
      "Only the brave dare attempt this challenge."
    ]
  };
  
  const options = verdicts[difficulty as keyof typeof verdicts] || verdicts.Medium;
  return options[Math.floor(Math.random() * options.length)];
}

function generatePitfalls(difficulty: string): string[] {
  const pitfalls = {
    Easy: [
      "Don't overthink the solution",
      "Watch out for edge cases with empty inputs",
      "Consider integer overflow for large numbers"
    ],
    Medium: [
      "Time complexity might be tricky to optimize",
      "Multiple approaches possible - choose wisely",
      "Edge cases can be subtle",
      "Memory usage might become a concern"
    ],
    Hard: [
      "Extremely complex edge cases",
      "Multiple algorithmic concepts combined",
      "Optimization is crucial for acceptance",
      "Implementation details matter significantly"
    ]
  };
  
  return pitfalls[difficulty as keyof typeof pitfalls] || pitfalls.Medium;
}

function generateSuggestions(difficulty: string): string[] {
  const suggestions = {
    Easy: [
      "Start with a brute force approach",
      "Look for simple patterns in the examples",
      "Consider built-in functions for common operations"
    ],
    Medium: [
      "Break the problem into smaller subproblems",
      "Consider different data structures",
      "Think about time-space tradeoffs",
      "Draw examples to understand the pattern"
    ],
    Hard: [
      "Study similar problems first",
      "Consider advanced algorithms and data structures",
      "Plan your approach carefully before coding",
      "Test with multiple complex examples"
    ]
  };
  
  return suggestions[difficulty as keyof typeof suggestions] || suggestions.Medium;
}
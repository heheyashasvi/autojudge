
export type DifficultyClass = 'Easy' | 'Medium' | 'Hard';

export interface ProblemData {
  title: string;
  description: string;
  inputDescription: string;
  outputDescription: string;
}

export interface PredictionResult {
  problemClass: DifficultyClass;
  problemScore: number;
  reasoning: string;
  keywords: string[];
  complexityAnalysis: {
    time: string;
    space: string;
    implementationEffort: number; // 1-10
    algorithmicDepth: number; // 1-10
  };
  verdict: string; // Witty AI persona comment
  pitfalls: string[];
  suggestions: string[];
}

export interface PredictionState {
  loading: boolean;
  error: string | null;
  result: PredictionResult | null;
}

export interface ChatMessage {
  role: 'user' | 'model';
  text: string;
}

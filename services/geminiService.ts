
import { GoogleGenAI, Type } from "@google/genai";
import { ProblemData, PredictionResult } from "../types";

export const getAI = () => new GoogleGenAI({ apiKey: process.env.API_KEY });

const RESPONSE_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    problemClass: { type: Type.STRING },
    problemScore: { type: Type.NUMBER },
    reasoning: { type: Type.STRING },
    keywords: { type: Type.ARRAY, items: { type: Type.STRING } },
    complexityAnalysis: {
      type: Type.OBJECT,
      properties: {
        time: { type: Type.STRING },
        space: { type: Type.STRING },
        implementationEffort: { type: Type.NUMBER },
        algorithmicDepth: { type: Type.NUMBER },
      },
      required: ["time", "space", "implementationEffort", "algorithmicDepth"]
    },
    verdict: { type: Type.STRING },
    pitfalls: { type: Type.ARRAY, items: { type: Type.STRING } },
    suggestions: { type: Type.ARRAY, items: { type: Type.STRING } },
  },
  required: ["problemClass", "problemScore", "reasoning", "keywords", "complexityAnalysis", "verdict", "pitfalls", "suggestions"],
};

export async function predictDifficulty(data: ProblemData): Promise<PredictionResult> {
  const ai = getAI();
  try {
    const prompt = `
      Act as "The Great Judge," a witty and expert competitive programming setter.
      Analyze this problem for a platform like Codeforces or LeetCode.
      
      Problem: ${data.title}
      Desc: ${data.description}
      Input: ${data.inputDescription}
      Output: ${data.outputDescription}

      In your "verdict", be slightly in character:
      - Easy: A bit bored, "A walk in the park for a toddler."
      - Medium: Respective, "Now we're talking. A solid test of logic."
      - Hard: Impressed, "May the gods of CPU cycles have mercy on the contestants."

      Return JSON following the strict schema.
    `;

    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: RESPONSE_SCHEMA,
      },
    });

    return JSON.parse(response.text) as PredictionResult;
  } catch (error) {
    console.error("Gemini API Error:", error);
    throw new Error("The Judge is currently in a meeting. Try again shortly.");
  }
}

export async function enrichPrediction(data: ProblemData, predictedClass: string, predictedScore: number): Promise<Partial<PredictionResult>> {
  const ai = getAI();
  try {
    const prompt = `
      Act as "The Great Judge," an expert competitive programming coach.
      
      I have analyzed a problem and determined it is: **${predictedClass}** (Score: ${predictedScore.toFixed(1)}/10).
      
      Problem: ${data.title}
      Desc: ${data.description}
      Input: ${data.inputDescription}
      Output: ${data.outputDescription}

      Please generate specific, insightful details for this problem to match my verdict.
      
      Return JSON with ONLY these fields:
      {
        "reasoning": "Brief explanation of why this specific problem fits the ${predictedClass} rating",
        "keywords": ["tag1", "tag2"],
        "complexityAnalysis": {
          "time": "Specific time complexity e.g. O(N log N)",
          "space": "Specific space complexity e.g. O(N)",
          "implementationEffort": 1-10 (based on lines of code/tricky cases),
          "algorithmicDepth": 1-10 (based on concept difficulty)
        },
        "verdict": "A witty one-liner verdict consistent with ${predictedClass}",
        "pitfalls": ["3 specific pitfalls for THIS problem"],
        "suggestions": ["3 specific ways to make THIS problem harder or optimize it"]
      }
    `;

    const ENRICH_SCHEMA = {
      type: Type.OBJECT,
      properties: {
        reasoning: { type: Type.STRING },
        keywords: { type: Type.ARRAY, items: { type: Type.STRING } },
        complexityAnalysis: {
          type: Type.OBJECT,
          properties: {
            time: { type: Type.STRING },
            space: { type: Type.STRING },
            implementationEffort: { type: Type.NUMBER },
            algorithmicDepth: { type: Type.NUMBER },
          },
          required: ["time", "space", "implementationEffort", "algorithmicDepth"]
        },
        verdict: { type: Type.STRING },
        pitfalls: { type: Type.ARRAY, items: { type: Type.STRING } },
        suggestions: { type: Type.ARRAY, items: { type: Type.STRING } },
      },
      required: ["reasoning", "keywords", "complexityAnalysis", "verdict", "pitfalls", "suggestions"],
    };

    const response = await ai.models.generateContent({
      model: "gemini-1.5-flash",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: ENRICH_SCHEMA,
      },
    });

    return JSON.parse(response.text) as Partial<PredictionResult>;
  } catch (error) {
    console.error("Gemini Enrichment Error:", error);
    // Fallback object will be handled by the caller or we can return null
    return {};
  }
}


import React from 'react';
import { PredictionResult } from '../types';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer } from 'recharts';

interface ResultDisplayProps {
  result: PredictionResult;
  onAskFollowUp: () => void;
}

const ResultDisplay: React.FC<ResultDisplayProps> = ({ result, onAskFollowUp }) => {
  const radarData = [
    { subject: 'Time', A: 80, fullMark: 100 },
    { subject: 'Space', A: 70, fullMark: 100 },
    { subject: 'Implementation', A: result.complexityAnalysis.implementationEffort * 10, fullMark: 100 },
    { subject: 'Algorithm', A: result.complexityAnalysis.algorithmicDepth * 10, fullMark: 100 },
    { subject: 'Abstraction', A: 60, fullMark: 100 },
  ];

  const classStyles = {
    'Easy': 'from-emerald-400 to-emerald-600',
    'Medium': 'from-amber-400 to-amber-600',
    'Hard': 'from-rose-400 to-rose-600',
  };

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-right-8 duration-500">
      {/* Hero Badge */}
      <div className={`p-8 rounded-[2.5rem] bg-gradient-to-br ${classStyles[result.problemClass]} text-white shadow-2xl relative overflow-hidden group`}>
        <div className="absolute -right-8 -bottom-8 opacity-10 group-hover:scale-110 transition-transform">
          <i className="fas fa-brain text-[12rem]"></i>
        </div>
        <div className="relative z-10">
          <div className="flex justify-between items-center mb-2">
            <span className="text-xs font-black tracking-widest uppercase opacity-75">Judge's Verdict</span>
            <div className="px-3 py-1 bg-white/20 rounded-full text-[10px] font-bold backdrop-blur-md">
              Score: {Math.round(result.problemScore)}
            </div>
          </div>
          <h2 className="text-5xl font-black mb-4 tracking-tighter">{result.problemClass}</h2>
          <p className="text-sm font-medium leading-relaxed italic opacity-90">
            "{result.verdict}"
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Radar Analysis */}
        <div className="bg-white rounded-3xl p-6 border border-gray-100 shadow-sm flex flex-col items-center">
          <h3 className="text-[10px] font-black uppercase text-gray-400 mb-4 tracking-widest w-full">Complexity Radar</h3>
          <div className="w-full h-48">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
                <PolarGrid stroke="#f1f5f9" />
                <PolarAngleAxis dataKey="subject" tick={{ fontSize: 10, fontWeight: 700, fill: '#94a3b8' }} />
                <Radar
                  name="Problem"
                  dataKey="A"
                  stroke="#6366f1"
                  fill="#6366f1"
                  fillOpacity={0.5}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Suggestion Box */}
        <div className="bg-gray-900 rounded-3xl p-6 text-white shadow-xl flex flex-col">
          <h3 className="text-[10px] font-black uppercase text-indigo-400 mb-4 tracking-widest">Judge's Notes</h3>
          <div className="flex-grow space-y-4">
            <div>
              <span className="text-[9px] font-bold text-gray-500 uppercase block mb-1">Common Pitfalls</span>
              <ul className="text-xs space-y-1">
                {result.pitfalls.map((p, i) => (
                  <li key={i} className="flex gap-2 items-start"><span className="text-rose-500">•</span> {p}</li>
                ))}
              </ul>
            </div>
            <div>
              <span className="text-[9px] font-bold text-gray-500 uppercase block mb-1">To Increase Difficulty</span>
              <ul className="text-xs space-y-1">
                {result.suggestions.map((s, i) => (
                  <li key={i} className="flex gap-2 items-start"><span className="text-indigo-400">•</span> {s}</li>
                ))}
              </ul>
            </div>
          </div>
          <button 
            onClick={onAskFollowUp}
            className="mt-6 w-full py-2 bg-indigo-600 hover:bg-indigo-500 rounded-xl text-[10px] font-black uppercase tracking-widest transition-colors"
          >
            Consult The Judge
          </button>
        </div>
      </div>

      {/* Reasoning */}
      <div className="bg-white rounded-3xl p-6 border border-gray-100 shadow-sm">
        <h3 className="text-[10px] font-black uppercase text-gray-400 mb-2 tracking-widest">Detailed Analysis</h3>
        <p className="text-xs text-gray-600 leading-relaxed font-medium mb-4">{result.reasoning}</p>
        <div className="flex gap-2 flex-wrap">
          {result.keywords.map((k, i) => (
            <span key={i} className="px-3 py-1 bg-indigo-50 text-indigo-600 text-[10px] font-bold rounded-full border border-indigo-100 uppercase tracking-tighter">
              {k}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ResultDisplay;

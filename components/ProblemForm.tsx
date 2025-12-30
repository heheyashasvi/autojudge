
import React from 'react';
import { ProblemData } from '../types';

interface ProblemFormProps {
  data: ProblemData;
  onChange: (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => void;
  onTemplateLoad: (template: ProblemData) => void;
  onSubmit: (e: React.FormEvent) => void;
  loading: boolean;
}

const TEMPLATES: Record<string, ProblemData> = {
  'Two Sum': {
    title: 'Two Sum',
    description: 'Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.',
    inputDescription: 'An array of integers and a single integer target.',
    outputDescription: 'Two indices of the numbers.'
  },
  'Dijkstra': {
    title: 'Shortest Path in Weighted Graph',
    description: 'Find the shortest path from node A to all other nodes in a graph with non-negative weights.',
    inputDescription: 'V vertices, E edges with weights.',
    outputDescription: 'List of shortest distances from source.'
  },
  'TSP': {
    title: 'Traveling Salesman',
    description: 'Find the shortest possible route that visits every city exactly once and returns to the origin city.',
    inputDescription: 'Adjacency matrix of distances between N cities.',
    outputDescription: 'Minimum weight of the tour.'
  }
};

const ProblemForm: React.FC<ProblemFormProps> = ({ data, onChange, onTemplateLoad, onSubmit, loading }) => {
  const wordCount = data.description.trim().split(/\s+/).filter(x => x).length;

  return (
    <div className="relative bg-white/80 backdrop-blur-xl rounded-3xl shadow-2xl border border-white/20 p-8 overflow-hidden">
      {loading && (
        <div className="absolute inset-0 z-10 pointer-events-none">
          <div className="h-full w-full bg-indigo-500/5 animate-pulse"></div>
          <div className="absolute top-0 left-0 w-full h-[2px] bg-indigo-500 shadow-[0_0_15px_#6366f1] animate-[scan_2s_linear_infinite]"></div>
        </div>
      )}

      <div className="flex justify-between items-center mb-6">
        <h2 className="text-xl font-black flex items-center gap-3">
          <span className="bg-indigo-600 text-white p-2 rounded-lg text-sm">
            <i className="fas fa-terminal"></i>
          </span>
          Problem Manifest
        </h2>
        <div className="flex gap-2">
          {Object.keys(TEMPLATES).map(name => (
            <button
              key={name}
              onClick={() => onTemplateLoad(TEMPLATES[name])}
              className="text-[10px] font-bold px-3 py-1 bg-gray-100 hover:bg-indigo-100 hover:text-indigo-600 rounded-full transition-all uppercase tracking-tighter"
            >
              {name}
            </button>
          ))}
        </div>
      </div>

      <form onSubmit={onSubmit} className="space-y-6">
        <div className="group">
          <label className="text-[10px] font-black uppercase text-gray-400 mb-1 block transition-colors group-focus-within:text-indigo-600">Title</label>
          <input
            name="title"
            value={data.title}
            onChange={onChange}
            className="w-full bg-gray-50/50 border-b-2 border-gray-100 focus:border-indigo-600 py-2 outline-none transition-all font-bold text-lg"
            placeholder="Name your challenge..."
          />
        </div>

        <div className="relative">
          <label className="text-[10px] font-black uppercase text-gray-400 mb-1 block">Description</label>
          <textarea
            name="description"
            value={data.description}
            onChange={onChange}
            rows={4}
            className="w-full bg-gray-50/50 border border-gray-100 rounded-xl p-4 focus:ring-2 focus:ring-indigo-600 outline-none transition-all resize-none text-sm"
            placeholder="Describe the task..."
          />
          <div className="absolute bottom-3 right-3 flex items-center gap-3">
             <span className="text-[10px] font-mono text-gray-400 bg-white px-2 py-0.5 rounded shadow-sm">
               {wordCount} WORDS
             </span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="text-[10px] font-black uppercase text-gray-400 mb-1 block">Input</label>
            <textarea
              name="inputDescription"
              value={data.inputDescription}
              onChange={onChange}
              rows={2}
              className="w-full bg-gray-50/50 border border-gray-100 rounded-xl p-3 focus:ring-2 focus:ring-indigo-600 outline-none transition-all text-xs"
            />
          </div>
          <div>
            <label className="text-[10px] font-black uppercase text-gray-400 mb-1 block">Output</label>
            <textarea
              name="outputDescription"
              value={data.outputDescription}
              onChange={onChange}
              rows={2}
              className="w-full bg-gray-50/50 border border-gray-100 rounded-xl p-3 focus:ring-2 focus:ring-indigo-600 outline-none transition-all text-xs"
            />
          </div>
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full py-4 bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-300 text-white rounded-2xl font-black text-sm uppercase tracking-widest shadow-xl shadow-indigo-200 hover:-translate-y-1 transition-all flex items-center justify-center gap-3"
        >
          {loading ? <i className="fas fa-spinner fa-spin"></i> : <i className="fas fa-bolt"></i>}
          Analyze Difficulty
        </button>
      </form>

      <style>{`
        @keyframes scan {
          0% { top: 0; }
          100% { top: 100%; }
        }
      `}</style>
    </div>
  );
};

export default ProblemForm;

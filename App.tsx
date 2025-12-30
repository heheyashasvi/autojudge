
import React, { useState, useRef, useEffect } from 'react';
import Header from './components/Header';
import ProblemForm from './components/ProblemForm';
import ResultDisplay from './components/ResultDisplay';
import { ProblemData, PredictionState, ChatMessage } from './types';
import { predictDifficulty, checkBackendHealth, getModelInfo } from './services/mlService';
import { getAI } from './services/geminiService';

const App: React.FC = () => {
  const [formData, setFormData] = useState<ProblemData>({
    title: '', description: '', inputDescription: '', outputDescription: ''
  });

  const [prediction, setPrediction] = useState<PredictionState>({
    loading: false, error: null, result: null
  });

  const [chatOpen, setChatOpen] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [userQuery, setUserQuery] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleTemplateLoad = (template: ProblemData) => {
    setFormData(template);
  };

  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    setPrediction({ loading: true, error: null, result: null });
    try {
      const result = await predictDifficulty(formData);
      setPrediction({ loading: false, error: null, result });
      setChatMessages([{ role: 'model', text: `I have judged your problem as ${result.problemClass}. What else would you like to know?` }]);
    } catch (err: any) {
      setPrediction({ loading: false, error: err.message, result: null });
    }
  };

  const handleChatSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!userQuery.trim() || chatLoading) return;

    const query = userQuery;
    setUserQuery('');
    setChatMessages(prev => [...prev, { role: 'user', text: query }]);
    setChatLoading(true);

    try {
      const ai = getAI();
      const chat = ai.chats.create({
        model: 'gemini-3-flash-preview',
        config: { systemInstruction: `You are the Great Judge who just evaluated this problem: ${formData.title}. Provide witty but helpful advice based on the evaluation result ${JSON.stringify(prediction.result)}.` }
      });
      const response = await chat.sendMessage({ message: query });
      setChatMessages(prev => [...prev, { role: 'model', text: response.text || 'The Judge remains silent.' }]);
    } catch (err) {
      setChatMessages(prev => [...prev, { role: 'model', text: 'Oops, lost my gavel. Try again.' }]);
    } finally {
      setChatLoading(false);
    }
  };

  return (
    <div className="flex flex-col min-h-screen bg-[#f8fafc] selection:bg-indigo-100 selection:text-indigo-700">
      <Header />
      
      <main className="flex-grow max-w-6xl mx-auto w-full p-6 md:py-12 relative">
        <div className="grid lg:grid-cols-2 gap-12">
          
          <div className="space-y-12">
            <header className="relative">
              <span className="inline-block px-3 py-1 bg-indigo-600 text-white text-[10px] font-black uppercase tracking-[0.2em] rounded mb-4">
                Powered by Custom ML Models
              </span>
              <h2 className="text-5xl font-black text-slate-900 tracking-tighter leading-[0.9]">
                Predictive <br />
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 to-violet-600">Problem Auditing</span>
              </h2>
              <p className="mt-6 text-slate-500 text-lg font-medium leading-relaxed max-w-md">
                Analyze algorithmic weight, implementation complexity, and mathematical depth in milliseconds.
              </p>
            </header>

            <ProblemForm 
              data={formData} 
              onChange={handleInputChange} 
              onTemplateLoad={handleTemplateLoad}
              onSubmit={handlePredict} 
              loading={prediction.loading}
            />
          </div>

          <div className="lg:sticky lg:top-24 h-fit">
            {prediction.error && (
              <div className="bg-rose-50 border-2 border-rose-100 p-6 rounded-3xl text-rose-600 flex items-center gap-4 animate-in zoom-in">
                <i className="fas fa-exclamation-circle text-2xl"></i>
                <p className="font-bold text-sm">{prediction.error}</p>
              </div>
            )}

            {!prediction.result && !prediction.loading && (
              <div className="bg-white/50 backdrop-blur rounded-[3rem] border-2 border-dashed border-slate-200 p-16 text-center">
                <div className="w-24 h-24 bg-white rounded-full shadow-lg flex items-center justify-center mx-auto mb-8 animate-bounce transition-all duration-1000">
                  <i className="fas fa-microscope text-slate-300 text-4xl"></i>
                </div>
                <h3 className="text-2xl font-black text-slate-800 mb-2 tracking-tight">System Idle</h3>
                <p className="text-slate-400 text-sm font-medium">Input a problem manifest to begin auditing.</p>
              </div>
            )}

            {prediction.loading && (
              <div className="bg-white rounded-[3rem] p-16 text-center shadow-xl border border-indigo-50">
                <div className="relative w-32 h-32 mx-auto mb-10">
                  <div className="absolute inset-0 border-8 border-indigo-50 rounded-full"></div>
                  <div className="absolute inset-0 border-8 border-indigo-600 rounded-full border-t-transparent animate-spin"></div>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <i className="fas fa-bolt text-indigo-600 text-4xl animate-pulse"></i>
                  </div>
                </div>
                <h3 className="text-2xl font-black text-slate-800 mb-2">Analyzing Complexity</h3>
                <p className="text-slate-400 text-xs font-bold uppercase tracking-widest">Running ML Models</p>
              </div>
            )}

            {prediction.result && (
              <ResultDisplay 
                result={prediction.result} 
                onAskFollowUp={() => setChatOpen(true)}
              />
            )}
          </div>
        </div>
      </main>

      {/* Judge's Consult Chat Drawer */}
      <div className={`fixed right-0 top-0 h-full w-full md:w-96 bg-white shadow-2xl z-[100] transform transition-transform duration-500 ease-in-out border-l border-gray-100 ${chatOpen ? 'translate-x-0' : 'translate-x-full'}`}>
        <div className="h-full flex flex-col">
          <div className="p-6 bg-slate-900 text-white flex justify-between items-center">
            <div>
              <h3 className="text-sm font-black uppercase tracking-widest">Consult The Judge</h3>
              <p className="text-[10px] text-indigo-400 font-bold uppercase">Private Session</p>
            </div>
            <button onClick={() => setChatOpen(false)} className="text-gray-400 hover:text-white transition-colors">
              <i className="fas fa-times"></i>
            </button>
          </div>
          
          <div className="flex-grow overflow-y-auto p-6 space-y-4">
            {chatMessages.map((m, i) => (
              <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[85%] p-3 rounded-2xl text-xs font-medium ${m.role === 'user' ? 'bg-indigo-600 text-white rounded-br-none' : 'bg-gray-100 text-slate-700 rounded-bl-none'}`}>
                  {m.text}
                </div>
              </div>
            ))}
            {chatLoading && (
              <div className="flex justify-start">
                <div className="bg-gray-100 p-3 rounded-2xl animate-pulse flex gap-2">
                  <div className="w-1 h-1 bg-gray-400 rounded-full animate-bounce"></div>
                  <div className="w-1 h-1 bg-gray-400 rounded-full animate-bounce [animation-delay:0.2s]"></div>
                  <div className="w-1 h-1 bg-gray-400 rounded-full animate-bounce [animation-delay:0.4s]"></div>
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          <form onSubmit={handleChatSubmit} className="p-4 border-t border-gray-100 flex gap-2">
            <input 
              value={userQuery}
              onChange={(e) => setUserQuery(e.target.value)}
              placeholder="Ask the judge..."
              className="flex-grow bg-gray-50 rounded-xl px-4 py-2 text-xs outline-none focus:ring-2 focus:ring-indigo-600 transition-all"
            />
            <button className="bg-indigo-600 text-white w-10 h-10 rounded-xl hover:bg-indigo-700 transition-all flex items-center justify-center shadow-lg shadow-indigo-100">
              <i className="fas fa-paper-plane text-xs"></i>
            </button>
          </form>
        </div>
      </div>
      
      {/* Overlay for chat */}
      {chatOpen && (
        <div 
          onClick={() => setChatOpen(false)} 
          className="fixed inset-0 bg-slate-900/20 backdrop-blur-sm z-[90] animate-in fade-in"
        />
      )}
    </div>
  );
};

export default App;

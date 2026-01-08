
import React from 'react';

const Header: React.FC = () => {
  return (
    <header className="bg-white border-b border-gray-200 py-4 px-6 sticky top-0 z-50 shadow-sm">
      <div className="max-w-6xl mx-auto flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="bg-indigo-600 p-2 rounded-lg">
            <i className="fas fa-gavel text-white text-xl"></i>
          </div>
          <div>
            <h1 className="text-xl font-bold text-gray-900 tracking-tight">AutoJudge</h1>
            <p className="text-xs text-gray-500 font-medium">Difficulty Prediction Engine</p>
          </div>
        </div>
        <nav className="hidden md:flex gap-6 items-center">
          <a href="https://github.com/heheyashasvi/autojudge#readme" target="_blank" rel="noopener noreferrer" className="text-sm font-semibold text-gray-600 hover:text-indigo-600 transition-colors">Documentation</a>
          <a href="https://github.com/heheyashasvi/autojudge/tree/main/backend/data" target="_blank" rel="noopener noreferrer" className="text-sm font-semibold text-gray-600 hover:text-indigo-600 transition-colors">Dataset</a>
          <a href="https://github.com/heheyashasvi/autojudge" target="_blank" rel="noopener noreferrer" className="bg-indigo-50 text-indigo-600 px-4 py-2 rounded-full text-sm font-bold hover:bg-indigo-100 transition-all">
            GitHub Repo
          </a>
        </nav>
      </div>
    </header>
  );
};

export default Header;

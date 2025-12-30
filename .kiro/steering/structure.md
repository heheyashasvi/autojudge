# Project Structure

## Root Level
```
├── App.tsx              # Main React application component
├── index.tsx            # React app entry point
├── index.html           # HTML template
├── types.ts             # Shared TypeScript type definitions
├── package.json         # Frontend dependencies and scripts
├── tsconfig.json        # TypeScript configuration
├── vite.config.ts       # Vite build configuration
└── .env.local           # Environment variables (API keys)
```

## Frontend Components
```
components/
├── Header.tsx           # Application header component
├── ProblemForm.tsx      # Problem input form component
└── ResultDisplay.tsx    # Prediction results display component
```

## Services Layer
```
services/
└── geminiService.ts     # Google Generative AI integration
```

## Backend Structure
```
backend/
├── app.py               # Flask API server
├── requirements.txt     # Python dependencies
├── setup_env.py         # Environment setup script
├── upload_dataset.py    # Dataset management utilities
├── ml/                  # Machine learning components
│   ├── __init__.py
│   ├── data_models.py   # Data structures and models
│   └── dataset_loader.py # Dataset loading utilities
├── data/                # Training data and datasets
│   ├── README.md
│   └── problems_data.jsonl
├── tests/               # Test suite
│   ├── __init__.py
│   └── test_data_models.py
└── venv/                # Python virtual environment
```

## Configuration & Tooling
```
.kiro/                   # Kiro IDE configuration
├── specs/               # Project specifications
└── steering/            # AI assistant guidance rules

.vscode/                 # VS Code configuration
```

## Architectural Patterns

### Component Organization
- **Separation of Concerns**: UI components, services, and types in separate directories
- **Single Responsibility**: Each component handles one specific UI concern
- **Type Safety**: Centralized type definitions in `types.ts`

### Backend Organization
- **Modular ML Pipeline**: ML components separated from API layer
- **Data Isolation**: Training data and models in dedicated directories
- **Test Coverage**: Comprehensive test suite with property-based testing

### API Design
- **RESTful Endpoints**: `/health` for monitoring, `/predict` for ML inference
- **JSON Communication**: Structured request/response format
- **Error Handling**: Consistent error response format across endpoints

### File Naming Conventions
- **React Components**: PascalCase (e.g., `ProblemForm.tsx`)
- **Services**: camelCase (e.g., `geminiService.ts`)
- **Python Modules**: snake_case (e.g., `data_models.py`)
- **Configuration**: lowercase with extensions (e.g., `tsconfig.json`)

### Import Patterns
- **Path Aliases**: Use `@/` for root-level imports in TypeScript
- **Relative Imports**: Prefer relative imports for nearby files
- **Absolute Imports**: Use absolute imports for external libraries
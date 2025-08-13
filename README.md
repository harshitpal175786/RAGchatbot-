# RAG Chatbot with PDF Processing

A Retrieval-Augmented Generation (RAG) chatbot that processes PDF documents and provides intelligent responses based on the document content.

## 🚀 Features

- **PDF Document Processing**: Extract and process text from PDF files
- **Vector Database**: Build and manage vector embeddings for efficient retrieval
- **RAG Implementation**: Generate contextual responses using document knowledge
- **GPU Support**: Optimized for GPU acceleration when available
- **Model Management**: Download and manage language models locally

## 📋 Prerequisites

- Python 3.8 or higher
- Git
- Sufficient disk space (at least 20GB for models and dependencies)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/harshitpal175786/RAGchatbot-.git
cd RAGchatbot
```

### 2. Create Python Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

### 3. Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install sentence-transformers
pip install chromadb
pip install pypdf
pip install langchain
pip install langchain-community
pip install accelerate
pip install bitsandbytes
pip install scikit-learn
pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
```

### 4. Download Required Models
```bash
# Run the model downloader
python scripts/download_model.py
```

## 📁 Project Structure

```
RAGchatbot/
├── scripts/
│   ├── build_vector_db.py      # Build vector database from documents
│   ├── download_model.py       # Download language models
│   ├── prepare_rag_data.py     # Prepare data for RAG processing
│   ├── data/                   # Document data directory
│   └── vectorstores/           # Vector database storage
├── models/                      # Downloaded model files
├── check_gpu.py                # GPU availability checker
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## 🚀 Usage

### 1. Check GPU Availability
```bash
python check_gpu.py
```

### 2. Prepare Your Documents
Place your PDF documents in the `scripts/data/` directory.

### 3. Build Vector Database
```bash
python scripts/build_vector_db.py
```

### 4. Prepare RAG Data
```bash
python scripts/prepare_rag_data.py
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the root directory:
```env
MODEL_PATH=./models
DATA_PATH=./scripts/data
VECTOR_DB_PATH=./scripts/vectorstores
```

### Model Settings
Edit `scripts/download_model.py` to change model configurations:
- Model size and type
- Download location
- Quantization settings

## 📊 Performance

- **CPU Mode**: Suitable for development and testing
- **GPU Mode**: Recommended for production use
- **Memory Usage**: Varies based on model size (2GB - 16GB)

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in scripts
   - Use smaller model variants
   - Enable gradient checkpointing

2. **Model Download Failures**
   - Check internet connection
   - Verify sufficient disk space
   - Try downloading during off-peak hours

3. **Import Errors**
   - Ensure virtual environment is activated
   - Reinstall dependencies: `pip install -r requirements.txt`

### Getting Help
- Check the logs in `scripts/retrieval_results.txt`
- Verify GPU drivers are up to date
- Ensure Python version compatibility

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Hugging Face for transformer models
- LangChain for RAG framework
- ChromaDB for vector storage
- PyTorch for deep learning framework

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section above
- Review the project documentation

---

**Note**: This project requires significant computational resources. Ensure you have adequate hardware before running large models.
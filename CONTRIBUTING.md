# Contributing to RLT Implementation

Thank you for your interest in contributing to the RLT (Reinforcement Learning Teachers) implementation! We welcome contributions from the community.

## 🤝 How to Contribute

### 1. Reporting Issues
- Use the GitHub issue tracker to report bugs
- Describe the issue clearly with steps to reproduce
- Include system information (OS, Python version, GPU model)
- Attach relevant logs or error messages

### 2. Suggesting Features
- Open a GitHub issue with the "enhancement" label
- Clearly describe the feature and its benefits
- Provide use cases and examples if possible

### 3. Code Contributions

#### Getting Started
1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Run tests: `pytest tests/`
5. Commit with clear messages: `git commit -m "Add: description of changes"`
6. Push to your fork: `git push origin feature/your-feature-name`
7. Create a Pull Request

#### Code Style
- Follow PEP 8 for Python code
- Use type hints where appropriate
- Add docstrings to all functions and classes
- Keep line length under 100 characters
- Use meaningful variable names

#### Testing
- Write tests for new features
- Ensure all existing tests pass
- Aim for >80% code coverage
- Include both unit and integration tests

### 4. Documentation
- Update README.md if adding new features
- Add docstrings to all new code
- Update relevant documentation files
- Include examples for new functionality

## 🎯 Priority Areas

We're particularly interested in contributions for:

### High Priority
- **Multi-GPU Support**: Implementing FSDP or DeepSpeed
- **Performance Benchmarks**: Comprehensive testing suite
- **Model Support**: Adding support for new architectures
- **Optimization**: Further memory and speed improvements

### Medium Priority
- **Visualization**: Training progress dashboards
- **Evaluation Metrics**: Additional reward components
- **Dataset Support**: More training datasets
- **Examples**: More notebook tutorials

### Nice to Have
- **Web Interface**: Simple UI for training
- **Model Serving**: Deployment utilities
- **Monitoring**: Integration with MLflow/W&B
- **Documentation**: Video tutorials

## 📝 Pull Request Process

1. **Before submitting:**
   - Ensure your code follows the style guide
   - All tests pass locally
   - Documentation is updated
   - Commit messages are clear

2. **PR Description should include:**
   - What changes were made
   - Why these changes are needed
   - Any breaking changes
   - Testing performed

3. **Review Process:**
   - PRs require at least one review
   - Address all feedback constructively
   - Keep PRs focused and reasonably sized

## 🧪 Testing Guidelines

```python
# Example test structure
def test_feature_name():
    """Test description."""
    # Arrange
    input_data = prepare_test_data()
    
    # Act
    result = your_function(input_data)
    
    # Assert
    assert result == expected_output
```

## 🏗️ Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/SLMtest.git
cd SLMtest

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Run linting
black src/ tests/
isort src/ tests/
```

## 💬 Communication

- **GitHub Issues**: For bugs and features
- **Discussions**: For general questions
- **Pull Requests**: For code contributions

## 🎖️ Recognition

Contributors will be:
- Listed in the CONTRIBUTORS.md file
- Mentioned in release notes
- Given credit in relevant documentation

## 📜 Code of Conduct

We follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/0/code_of_conduct/). Please be respectful and constructive in all interactions.

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for helping make RLT implementation better! 🚀
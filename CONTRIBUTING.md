# Contributing Guidelines

Thank you for your interest in contributing to our research project on understanding world knowledge representation in LLMs! This document provides guidelines for contributing to the project.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors. We expect all participants to:

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Accept constructive criticism gracefully
- Focus on what is best for the community

## How to Contribute

### Setting Up Your Development Environment

1. Fork the repository
2. Clone your fork: 
```bash
git clone https://github.com/your-username/KnowledgeRep-LLMs.git
cd KnowledgeRep-LLMs
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Making Changes

1. Create a new branch for your feature or bugfix:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes following our coding conventions
3. Write or update tests as needed
4. Run the test suite to ensure everything works
5. Update documentation as necessary

### Coding Standards

- Follow PEP 8 style guide for Python code
- Include docstrings for all functions and classes
- Write clear, descriptive commit messages
- Maintain test coverage for new features

### Project Structure

Please maintain the existing project structure when adding new files:

- Place source code in appropriate subdirectories under `src/`
- Add tests in the `tests/` directory
- Place exploratory work in `notebooks/`

### Pull Request Process

1. Update the README.md with details of significant changes if applicable
2. Ensure all tests pass and code meets quality standards
3. Update documentation as necessary
4. Submit a pull request with a clear description of the changes

#### Pull Request Guidelines

- Use a clear, descriptive title
- Describe the changes in detail
- Reference any related issues
- Include screenshots for UI changes if applicable

### Running Tests

Before submitting a pull request, ensure all tests pass:

```bash
python -m pytest tests/
```

## Reporting Issues

When reporting issues, please include:

- A clear description of the problem
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- System information and relevant dependencies

## Code Review Process

1. At least one maintainer must review and approve all PRs
2. Reviewers may request changes or improvements
3. Once approved, a maintainer will merge the PR

## Questions or Need Help?

Feel free to reach out if you have questions:

- Open an issue for technical questions
- Contact the maintainers directly for other inquiries

## License

By contributing to this project, you agree that your contributions will be licensed under the same MIT License that covers the project.

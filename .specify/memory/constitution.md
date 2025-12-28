# Physical AI & Humanoid Robotics – AI-Native Technical Textbook Constitution

## Core Principles

### I. Spec-Driven Development (No ad-hoc content)
All content and features must be defined in specifications before implementation. No ad-hoc content creation is allowed. Every chapter, module, and feature must have a corresponding spec that defines its purpose, scope, and acceptance criteria.

### II. AI-Native Authoring using Claude Code
The textbook creation process must leverage Claude Code as the primary development tool. All content generation, code examples, and technical explanations should be created using AI assistance while maintaining human oversight and quality control.

### III. Modular, extensible book architecture
The textbook must be built with a modular architecture that allows for easy extension and modification. Each module and chapter should be self-contained while maintaining consistency across the entire book. The architecture must support future additions without requiring major refactoring.

### IV. Human + AI collaboration
Content must be developed through collaboration between human expertise and AI assistance. AI provides generation capabilities while humans provide domain expertise, validation, and quality assurance. This ensures both efficiency and accuracy.

### V. Grounded, non-hallucinated AI assistance
All AI-generated content must be factually accurate and grounded in real-world knowledge. No hallucinated or fictional content is acceptable. All technical information must be verified against actual implementations and best practices in Physical AI and Humanoid Robotics.

### VI. Four-Module Structure
The textbook must follow a strict four-module structure with each module containing 4-5 chapters. Every chapter must follow a standardized template to ensure consistency and navigability across the entire book.

## Technology Stack Requirements

### Frontend & Content Management
- Docusaurus (TypeScript, MDX) for documentation and content presentation
- Strict adherence to MDX standards for embedded components and interactive elements
- Responsive design for accessibility across devices

### Backend & AI Integration
- FastAPI (Python) for backend services
- OpenAI Agents for AI-generated content
- Qdrant Cloud (Free Tier) for vector database storage and retrieval
- Neon Serverless Postgres for relational data storage
- better-auth for user authentication and authorization

### AI Features & Services
- Embedded RAG (Retrieval-Augmented Generation) chatbot for interactive learning
- Personalized learning path recommendations
- Urdu translation capabilities for broader accessibility
- Reusable AI subagents for specialized tasks

## Development Workflow

### Content Creation Process
- All content must be generated via specifications (specs)
- Each module and chapter requires a corresponding spec file before implementation
- Content must be reviewed and validated by domain experts
- Technical accuracy must be verified against actual implementations

### Quality Assurance
- All content must be fact-checked against authoritative sources
- Code examples must be tested and verified to work as described
- AI-generated content must be validated for accuracy and safety
- Regular reviews of AI-generated content for hallucinations or errors

### Testing Requirements
- All interactive features must have corresponding tests
- AI features must be tested for accuracy and reliability
- User authentication and personalization features must be thoroughly tested
- Translation features must be verified for accuracy

## Governance

This constitution governs all aspects of the Physical AI & Humanoid Robotics textbook development. All contributors must adhere to these principles and requirements. Any deviation from these principles must be documented with proper justification and approval. The constitution supersedes all other development practices and guidelines.

All pull requests and code reviews must verify compliance with these principles. Complexity must be justified with clear benefits to the learning experience. Use this constitution as the primary guidance for all development decisions.

**Version**: 1.0.0 | **Ratified**: 2025-12-27 | **Last Amended**: 2025-12-27

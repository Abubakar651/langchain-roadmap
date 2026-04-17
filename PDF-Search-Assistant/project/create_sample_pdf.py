"""
Creates a sample PDF for testing the PDF Search Assistant.
Requires: pip install reportlab

Run: python create_sample_pdf.py
"""

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    import os

    output_path = os.path.join(os.path.dirname(__file__), "..", "sample_docs", "ai_overview.pdf")
    output_path = os.path.abspath(output_path)

    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    content = [
        ("Introduction to Artificial Intelligence", "h1"),
        ("Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn. The term was coined by John McCarthy in 1956 at the Dartmouth Conference, which is considered the birthplace of AI as a field.", "body"),
        ("Machine Learning", "h2"),
        ("Machine learning is a subset of AI that enables computers to learn from data without being explicitly programmed. There are three main types: supervised learning, unsupervised learning, and reinforcement learning. Popular algorithms include linear regression, decision trees, random forests, and neural networks.", "body"),
        ("Deep Learning and Neural Networks", "h2"),
        ("Deep learning uses artificial neural networks with multiple layers (hence 'deep') to learn representations of data. Convolutional Neural Networks (CNNs) excel at image recognition. Recurrent Neural Networks (RNNs) handle sequential data. Transformers, introduced in 2017, revolutionised natural language processing.", "body"),
        ("Large Language Models", "h2"),
        ("Large Language Models (LLMs) are transformer-based models trained on massive text datasets. GPT-4 by OpenAI, Claude by Anthropic, LLaMA by Meta, and Gemini by Google are leading examples. These models can generate text, answer questions, write code, and perform complex reasoning tasks.", "body"),
        ("Retrieval-Augmented Generation (RAG)", "h2"),
        ("RAG is a technique that combines information retrieval with text generation. Instead of relying solely on training data, RAG systems first retrieve relevant documents from a knowledge base, then use an LLM to generate answers grounded in those documents. This reduces hallucinations and enables use of up-to-date information.", "body"),
        ("Vector Databases", "h2"),
        ("Vector databases store high-dimensional embedding vectors and support efficient similarity search. Unlike traditional databases that match exact values, vector databases find semantically similar content. FAISS by Meta, Pinecone, ChromaDB, and Weaviate are popular choices. They are the backbone of modern RAG systems.", "body"),
        ("Applications of AI", "h2"),
        ("AI is transforming numerous industries. In healthcare, AI assists in medical imaging diagnosis and drug discovery. In finance, it powers fraud detection and algorithmic trading. Self-driving cars use computer vision and reinforcement learning. Virtual assistants like Siri and Alexa use natural language processing. Content recommendation systems on Netflix and Spotify use collaborative filtering.", "body"),
        ("Ethical Considerations", "h2"),
        ("AI raises important ethical questions around bias, privacy, transparency, and accountability. Algorithmic bias can perpetuate discrimination if training data reflects historical inequalities. Explainable AI (XAI) aims to make model decisions interpretable. Responsible AI development requires diverse teams and robust testing across demographic groups.", "body"),
        ("The Future of AI", "h2"),
        ("Artificial General Intelligence (AGI) refers to AI that matches human-level intelligence across all domains — still an open research problem. Current AI systems are narrow, excelling in specific tasks. Multimodal AI models that process text, images, audio, and video simultaneously represent the near-term frontier. AI safety research focuses on aligning AI goals with human values.", "body"),
    ]

    for text, style_name in content:
        if style_name == "h1":
            story.append(Paragraph(text, styles["Title"]))
        elif style_name == "h2":
            story.append(Spacer(1, 0.2 * inch))
            story.append(Paragraph(text, styles["Heading2"]))
        else:
            story.append(Paragraph(text, styles["BodyText"]))
        story.append(Spacer(1, 0.1 * inch))

    doc.build(story)
    print(f"Sample PDF created: {output_path}")
    print("You can now upload this PDF in the Streamlit app.")

except ImportError:
    print("reportlab not installed. Install with: pip install reportlab")
    print("Alternatively, use any existing PDF file with the app.")

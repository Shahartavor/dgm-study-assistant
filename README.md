# Deep Generative Models Study Assistant

DGM Study Assistant is a Retrieval-Augmented Generation (RAG) system designed to help students learn Deep Generative Models.
It retrieves information from the course transcript, academic papers, and other relevant knowledge sources, and provides an interactive Gradio-based chatbot interface.
The assistant can explain concepts, answer questions, and help with code examples related to:
* Expectation–Maximization (EM)
* Gaussian Mixture Models (GMM)
* Variational Autoencoders (VAEs)
* PixelCNN
* Normalizing Flows
* Diffusion Models
* Monte Carlo Methods
* And additional topics taught throughout the course

<img width="700" height="700" alt="dgm_flow" src="https://github.com/user-attachments/assets/ca7fad15-6c3f-4271-a94c-34f91af839f4" />


## Installation 
**Install uv**

uv is a fast Python package and environment manager that replaces pip, venv, and pip-tools.
macOS & Linux: 

`curl -LsSf https://astral.sh/uv/install.sh | sh`

Then reload your shell: 

`source ~/.zshrc  # or ~/.bashrc`

macOS (Homebrew alternative):

`brew install uv`

Windows (PowerShell):

`powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`

Verify Installation:

`uv --version`

**Create and activate virtual environment**
```
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

**Install project**

`uv pip install -e .`

**Copy environment template**

`cp .env.example .env`

**Run Gradio app**

`uv run ui/app.py`

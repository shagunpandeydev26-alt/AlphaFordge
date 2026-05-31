# Setup Guide

This guide will help you set up your environment and download the necessary models for the RL Trading Agent project.

## 1. Clone the Repository

```sh
git clone https://github.com/yourusername/RLTradingAgent.git
cd RLTradingAgent
```

## 2. Set Up a Virtual Environment

It is recommended to use a virtual environment to manage dependencies.

### On macOS/Linux:
```sh
python3 -m venv venv
source venv/bin/activate
```

### On Windows:
```powershell
python -m venv venv
venv\Scripts\activate
```

## 3. Install Dependencies

Once the virtual environment is activated, install the required dependencies:

```sh
pip install -r requirements.txt
```

## 4. Download Pretrained Models

Pretrained models are hosted on GitHub Releases. Download them using the following steps:

1. Go to the [Releases](https://github.com/Aryan10/RLTradingAgent/releases) page.
2. Download the latest model files (`*.zip`).
3. Extract the files into the `models/` directory:

   ```sh
   unzip path/to/downloaded/model.zip -d models/
   ```

## 5. Verify Installation

To check if everything is set up correctly, try running:

```sh
python src/ui/app.py
```

If the Streamlit app starts without errors, the setup is complete!

---

For further issues, open an issue in the repository.


# Langchain Agents

This repository contains a Python script that demonstrates how to build and use Langchain agents with various tools. The agent is capable of fetching stock prices, getting the current temperature of a city, retrieving currency exchange rates, and searching for YouTube videos.

## Features

- **Modular Design:** The script is structured into clear steps, from setting up the environment to running the agent with memory.
- **Custom Tools:** It includes custom tools for:
    - `get_stock_price`: Fetches the stock price for a given ticker symbol using `yfinance`.
    - `get_temperature`: Retrieves the current temperature for a specified city using the WeatherAPI.
    - `get_currency_exchange_rates`: Gets currency exchange rates using the ExchangeRate-API.
    - `YouTubeSearchTool`: Searches for YouTube videos.
- **LLM Integration:** Utilizes Google's Gemini Pro as the language model.
- **Agent with Memory:** Demonstrates how to create a conversational agent that can remember previous interactions.

## Prerequisites

- Python 3.10 or higher
- An environment that supports `pip` for package installation.

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/bhoga01ai/langchain-agents.git
   cd langchain-agents
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your environment variables:**
   - Create a `.env` file in the root of the project.
   - Add your API keys to the `.env` file:
     ```
     GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
     weatherAPIKey="YOUR_WEATHER_API_KEY"
     ```

## Usage

To run the agent, execute the following command in your terminal:

```bash
python langchain_tools_agent.py
```

The script will then:
1. Initialize the language model.
2. Set up the custom tools.
3. Run a series of predefined queries to demonstrate the agent's capabilities.
4. Showcase the agent's ability to retain information in a conversation.

Feel free to modify the `langchain_tools_agent.py` file to experiment with different queries or to extend the agent's functionality.

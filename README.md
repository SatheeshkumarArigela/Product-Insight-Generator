# AI-Powered Product Insight Generator

An LLM-powered application that transforms customer product reviews into structured insights using **Google Gemini, Python, and Streamlit**. It analyzes customer sentiment, identifies product strengths and pain points, summarizes recurring themes, and generates actionable recommendations for sellers.

## Overview

Customer reviews contain valuable feedback, but manually analyzing large amounts of text can be time-consuming. This application automates the process using a three-stage workflow that combines LLM-based analysis with Python-based aggregation.

Users can upload a CSV or TXT file, paste reviews directly, or explore the application using sample reviews.

## Key Features

- **Sentiment Analysis:** Classifies reviews as positive, neutral, or negative.
- **Aspect-Based Analysis:** Identifies topics such as battery life, camera quality, pricing, performance, delivery, and customer support.
- **Pain Points and Praises:** Extracts customer complaints and positive feedback.
- **Automated Insight Reports:** Generates summaries, product strengths, key issues, and recommendations for sellers.
- **Visual Analytics:** Displays sentiment percentages and charts of positive and negative aspect-level feedback.
- **Flexible Input:** Supports CSV uploads, TXT uploads, and manually pasted reviews.
- **Structured LLM Outputs:** Uses JSON responses to organize extracted information and generated reports.
- **Error Handling:** Includes JSON parsing, response validation, retries, and user-facing error messages.
- **Review-Count-Aware Summaries:** Adjusts the summary length based on the number of reviews analyzed.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application logic and data processing |
| Google Gemini API | Review analysis and insight generation |
| Streamlit | Interactive web interface |
| pandas | CSV reading and data aggregation |
| JSON | Structured LLM responses |
| Python `Counter` and `defaultdict` | Sentiment, aspect, praise, and complaint aggregation |

## How It Works

The application follows a three-stage processing pipeline.

### 1. Extract — LLM-Based Analysis

Customer reviews are divided into batches. Google Gemini analyzes each batch and returns structured JSON containing:

- Overall sentiment
- Product aspects and their sentiment
- Customer pain points
- Customer praises

Each review is limited to 500 characters during extraction to control input size.

### 2. Aggregate — Python-Based Processing

Python processes the extracted results without making an additional LLM call for aggregation.

It calculates:

- Positive, neutral, and negative sentiment counts
- Sentiment counts for product aspects
- Frequently occurring complaints and praises
- The most frequently discussed product aspects

### 3. Synthesize — LLM-Based Reporting

The aggregated statistics and representative feedback are passed to Google Gemini to generate a final report containing:

- Headline and overall summary
- Overall sentiment verdict
- Top product strengths and supporting evidence
- Key issues and severity levels
- Recommendations for sellers
- Guidance on who should buy or avoid the product
- A confidence note based on the review count

The summary length changes according to the number of reviews processed.

## Application Workflow

1. Enter an optional product name.
2. Upload a CSV or TXT file, paste reviews, or use the provided sample reviews.
3. Click **Generate insights**.
4. The application extracts review-level information using Gemini.
5. Python aggregates sentiment and aspect statistics.
6. Gemini generates the final product insight report.
7. Explore sentiment metrics, aspect charts, strengths, issues, and recommendations.
8. Expand the prompt-chain section to inspect sample extraction results, aggregated statistics, and the final report.

## Project Structure

```text
ai-product-insight-generator/
├── app.py
├── requirements.txt
├── README.md
└── .streamlit/
    └── secrets.toml   # Optional local configuration
```

Do not upload `secrets.toml` or any file containing your API key to GitHub.

## Prerequisites

- Python 3.10 or a compatible Python version supported by your installed dependencies
- A Google Gemini API key
- pip package manager

## Installation and Setup

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd ai-product-insight-generator
```

Replace the repository URL with your actual GitHub repository URL.

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Gemini API Key

Set your API key as an environment variable.

**Windows PowerShell:**

```powershell
$env:GEMINI_API_KEY="your_api_key_here"
```

**macOS or Linux:**

```bash
export GEMINI_API_KEY="your_api_key_here"
```

Alternatively, for local Streamlit configuration, create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your_api_key_here"
```

Never commit your API key to source control. For deployment, configure the key using the hosting platform's secret-management settings.

### 5. Run the Application

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open it in your browser to use the application.

## Input Requirements and Limits

- Supported file formats: CSV and TXT.
- CSV files should contain a review column. If no column name contains `review`, the application uses the first column.
- TXT files should contain one review per line.
- At least **3 reviews** are required.
- The application processes a maximum of **60 reviews** per run.
- Reviews are processed in batches of **15**.
- Each review is truncated to a maximum of **500 characters** during extraction.

These limits are configured in the application to help manage API usage and rate limits.

## Environment Configuration

The application uses the following environment variables:

| Variable | Purpose | Default |
|---|---|---|
| `GEMINI_API_KEY` | Authenticates requests to Google Gemini | No default |
| `INSIGHT_MODEL` | Selects the Gemini model used by the application | `gemini-3.5-flash-lite` |

The model can be changed by setting `INSIGHT_MODEL` to a model identifier available to your Google AI account.

## Dependencies

The project uses the following packages, listed in `requirements.txt`:

```text
streamlit
google-genai
pandas
```

## Error Handling and Reliability

The application includes several mechanisms to handle common API and response issues:

- Retries LLM requests when calls fail or JSON responses are invalid.
- Validates that required keys are present in the generated JSON.
- Displays user-facing error messages when processing fails.
- Adds delays between API calls to help manage rate limits.
- Uses a low-temperature generation configuration for more consistent structured outputs.

These measures improve robustness but do not guarantee that every generated insight will be accurate.

## Current Limitations

- Analysis quality depends on the quality and representativeness of the supplied reviews.
- LLM-generated classifications and recommendations may contain errors.
- The application processes at most 60 reviews per run.
- It does not currently include a persistent database or historical-report storage.
- No automated accuracy benchmark or formal evaluation dataset is included in the current implementation.
- API quotas and model availability depend on the Google Gemini service and account configuration.

## Potential Future Improvements

- Add automated evaluation using a manually labelled review dataset.
- Support larger datasets through asynchronous or background processing.
- Export insight reports to PDF or CSV.
- Add filters for products, sentiment, and review categories.
- Introduce caching to reduce repeated API calls.
- Add unit tests for JSON parsing, aggregation, and report validation.
- Track changes in customer sentiment across multiple review uploads.

## Learning Outcomes

This personal project explores practical GenAI application development, including:

- Prompt engineering and multi-stage prompt chaining
- Structured JSON generation and validation
- LLM API integration
- Combining deterministic Python logic with LLM reasoning
- Text preprocessing and sentiment aggregation
- API error handling and rate-limit awareness
- Building interactive AI applications with Streamlit

## Disclaimer

This project is intended for educational and demonstration purposes. Generated insights should be reviewed before being used for business decisions. Use sample, public, or appropriately authorized customer reviews.

## Author

**Satheesh Kumar Arigela**

- **GitHub:** [SatheeshkumarArigela](https://github.com/SatheeshkumarArigela)
- **LinkedIn:** [Arigela Satheesh Kumar](https://www.linkedin.com/in/arigela-satheesh-kumar-59a00822b/)


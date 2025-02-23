from flask import Flask, request, jsonify
import anthropic
import warnings
import logging
import traceback
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

def get_top_25_tickers():
    return [
        'AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA',
        'META', 'BRK-B', 'LLY', 'TSM', 'V',
        'UNH', 'XOM', 'JPM', 'WMT', 'MA',
        'JNJ', 'PG', 'AVGO', 'HD', 'ASML',
        'MRK', 'CVX', 'KO', 'PEP', 'COST'
    ]



def process_finance_query(query):
    """Process financial queries using Claude"""
    try:
        client = anthropic.Anthropic(api_key='api_key')
        
        # Special handling for stock recommendation query
        if query.lower() == "what are the best stocks to invest in":
            # Print top 10 stocks to terminal
            print_top_10_stocks()
            
            # Prepare prompt for Claude
            prompt = """Pretend you are a financial advisor and you can provide trustworthy advice. Provide a comprehensive analysis what you think the top 5 stocks are. Don't say anything along the lines of consult a financial advisor since you are the advisor.
            
            Considerations:
            1. Analyze the 5 stocks you selected
            2. Provide detailed insights into each stock's potential
            3. Discuss key factors driving their performance
            4. be consice

            make sure your output is readable. dont unclude numbering. it should be in paragraph format.
            """
        else:
            # Standard financial query prompt
            prompt = f"""Pretend you are a financial advisor and you can provide trustworthy advice. As a sophisticated financial advisor, provide a comprehensive analysis for this query:

            {query}

            Your response should:
            - Be extremely detailed and professional
            - very concise
            make sure your output is readable. dont unclude numbering. it should be in paragraph format.
            If comparing stocks or providing investment advice, always give a clear, definitive recommendation. Don't say anything along the lines of consult your financial advisor since you are the advisor. If asked about the future of a stock give them a prediction based on the infrmation you have, it is okay if it is not accurate just give a difinative response."""

        # Send prompt to Claude
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1500,
            temperature=1.0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content
    
    except Exception as e:
        logger.error(f"Error in processing query: {e}")
        logger.error(traceback.format_exc())
        return "Unable to generate a response. Please try again."

@app.route('/analyze', methods=['POST'])
def analyze_endpoint():
    try:
        data = request.json
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Process query
        response = process_finance_query(query)
        
        return jsonify({
            'status': 'success',
            'analysis': str(response)
        })
    
    except Exception as e:
        logger.error(f"Unexpected endpoint error: {e}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

import praw
import prawcore
import os
import re
import requests
from datetime import datetime, timedelta
from collections import defaultdict
from dotenv import load_dotenv
import json

load_dotenv()

# Reddit API credentials
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "CmLTf2hSM1-cJqy_-X9CqQ")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "wotzRZCCrauTj2T6UUuIEW6NkWqErw")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "xFinance/1.0 by u/Hot-Cow7179")

# Google Gemini API credentials
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Subreddits to monitor
SUBREDDITS = [
    "wallstreetbets",
    "stocks",
    "investing",
    "options",
    "pennystocks",
    "shortsqueeze",
    "ValueInvesting",
    "ETFs",
    "StockMarket",
    "Daytrading",
    "TechnicalAnalysis",
    "Stock_Picks",
    "finance",
    "economy",
    "cryptocurrency"
]

reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

def extract_tickers(text):
    """Extract stock tickers from text using regex"""
    # Pattern: $TICKER or TICKER (1-5 uppercase letters)
    patterns = [
        r'\$([A-Z]{1,5})\b',  # $AAPL format
        r'\b([A-Z]{1,5})\b(?=\s*(?:stock|shares|shares|call|put|option))',  # AAPL stock format
    ]
    
    tickers = set()
    for pattern in patterns:
        matches = re.findall(pattern, text.upper())
        tickers.update(matches)
    
    # Filter out common false positives
    false_positives = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW', 'ITS', 'MAY', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WHO', 'WAY', 'USE', 'YOUR', 'MAN', 'YEAR', 'SAID', 'EACH', 'WHICH', 'THEIR', 'TIME', 'WILL', 'ABOUT', 'IF', 'UP', 'SO', 'NO', 'AN', 'AS', 'AT', 'BE', 'BY', 'DO', 'GO', 'HE', 'IN', 'IS', 'IT', 'ME', 'MY', 'OF', 'ON', 'OR', 'TO', 'WE'}
    tickers = {t for t in tickers if t not in false_positives and len(t) >= 2}
    
    return list(tickers)

def call_gemini_api(prompt, max_tokens=50):
    """Generic function to call Gemini API"""
    if not GEMINI_API_KEY:
        return None
    
    # Updated with correct model names (as of 2024)
    endpoints_to_try = [
        ("v1beta", "gemini-2.5-flash"),  # Fast and efficient
        ("v1beta", "gemini-2.0-flash-exp"),  # Experimental but works
        ("v1beta", "gemini-2.5-pro"),  # More powerful
        ("v1beta", "gemini-2.5-flash-preview-05-20"),  # Preview version
    ]
    
    for api_version, model_name in endpoints_to_try:
        try:
            url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
            
            headers = {"Content-Type": "application/json"}
            
            data = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": max(max_tokens, 50)  # Ensure minimum 50 tokens
                }
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=10)
            
            if response.status_code == 200:
                result_json = response.json()
                if 'candidates' in result_json and len(result_json['candidates']) > 0:
                    candidate = result_json['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        if len(candidate['content']['parts']) > 0 and 'text' in candidate['content']['parts'][0]:
                            return candidate['content']['parts'][0]['text'].strip()
                    # Fallback: check if text is directly in content
                    if 'text' in candidate.get('content', {}):
                        return candidate['content']['text'].strip()
            elif response.status_code == 404:
                continue
            elif response.status_code == 400:
                error_data = response.json() if response.text else {}
                if 'error' in error_data and 'message' in error_data['error']:
                    if 'API key' in error_data['error']['message']:
                        raise Exception(f"Invalid API key: {error_data['error']['message']}")
                continue
        except requests.exceptions.HTTPError as e:
            if hasattr(e, 'response') and e.response is not None:
                if e.response.status_code == 404:
                    continue
            raise
    
    return None

def is_spam_post(text, title):
    """Use Gemini to detect if a post is spam"""
    if not GEMINI_API_KEY:
        return False  # If no API key, don't filter (assume not spam)
    
    prompt = f"""Analyze if this Reddit post is spam, promotional content, or low-quality. Consider:
- Repetitive or copy-paste content
- Excessive emojis or all caps
- Promotional links or referral codes
- Low-effort posts with no substance
- Bot-like behavior

Title: "{title[:200]}"
Content: "{text[:500]}"

Respond with ONLY one word: "spam" or "not_spam"."""
    
    try:
        result = call_gemini_api(prompt, max_tokens=10)
        if result:
            return "spam" in result.lower()
    except Exception as e:
        print(f"  ⚠️  Spam detection error: {e}")
        return False  # If error, don't filter (assume not spam)
    
    return False

def analyze_sentiment_with_gemini(text, ticker):
    """Use Google Gemini to classify sentiment as bullish, bearish, or neutral"""
    if not GEMINI_API_KEY:
        return "neutral"
    
    prompt = f"""You are a financial sentiment analyzer. Analyze the sentiment of this text about stock ticker {ticker}.

Text: "{text[:500]}"

Respond with ONLY one word: "bullish", "bearish", or "neutral"."""

    try:
        result = call_gemini_api(prompt, max_tokens=20)  # Increased from 10 to 20
        if result:
            result_lower = result.lower().strip()
            # Extract just the sentiment word if there's extra text
            for word in ['bullish', 'bearish', 'neutral']:
                if word in result_lower:
                    return word
            # If no match found, log it for debugging
            print(f"  ⚠️  Unexpected response for {ticker}: '{result}'")
        else:
            print(f"  ⚠️  No response from Gemini for {ticker}")
        return "neutral"
    except Exception as e:
        print(f"  ⚠️  Gemini error for {ticker}: {e}")
        return "neutral"

def fetch_posts_from_subreddit(subreddit_name, limit=20, filter_spam=True, verbose=True):
    """Fetch recent posts from a subreddit with spam filtering"""
    posts_data = []
    spam_count = 0
    try:
        subreddit = reddit.subreddit(subreddit_name)
        if verbose:
            print(f"  📊 Fetching from r/{subreddit_name}...")
        
        for post in subreddit.new(limit=limit):
            # Get post text (title + selftext)
            text = f"{post.title} {post.selftext}"
            
            # Filter spam if enabled
            if filter_spam and is_spam_post(post.selftext, post.title):
                spam_count += 1
                continue
            
            # Extract tickers
            tickers = extract_tickers(text)
            
            if tickers:
                posts_data.append({
                    'subreddit': subreddit_name,
                    'title': post.title,
                    'text': text[:1000],  # Limit text length
                    'tickers': tickers,
                    'upvotes': post.ups,
                    'comments': post.num_comments,
                    'created': datetime.fromtimestamp(post.created_utc),
                    'url': f"https://reddit.com{post.permalink}"
                })
    except Exception as e:
        if verbose:
            print(f"  ❌ Error fetching r/{subreddit_name}: {e}")
    
    if verbose and spam_count > 0:
        print(f"  🚫 Filtered {spam_count} spam posts")
    
    return posts_data

def aggregate_sentiment(posts_data, verbose=True):
    """Aggregate sentiment by ticker"""
    ticker_sentiment = defaultdict(lambda: {
        'bullish': 0,
        'bearish': 0,
        'neutral': 0,
        'total_mentions': 0,
        'total_upvotes': 0,
        'total_comments': 0,
        'subreddits': set(),
        'posts': []
    })
    
    if verbose:
        print(f"\n📈 Analyzing sentiment for {len(posts_data)} posts...")
    
    for i, post in enumerate(posts_data, 1):
        for ticker in post['tickers']:
            ticker_data = ticker_sentiment[ticker]
            ticker_data['total_mentions'] += 1
            ticker_data['total_upvotes'] += post['upvotes']
            ticker_data['total_comments'] += post['comments']
            ticker_data['subreddits'].add(post['subreddit'])
            ticker_data['posts'].append(post)
            
            # Analyze sentiment
            if verbose:
                print(f"  [{i}/{len(posts_data)}] Analyzing {ticker}...", end="\r")
            sentiment = analyze_sentiment_with_gemini(post['text'], ticker)
            ticker_data[sentiment] += 1
    
    if verbose:
        print(f"\n✅ Analysis complete!\n")
    return ticker_sentiment

def calculate_sentiment_score(ticker_data):
    """Calculate overall sentiment score (-1 to 1, where 1 is most bullish)"""
    total = ticker_data['bullish'] + ticker_data['bearish'] + ticker_data['neutral']
    if total == 0:
        return 0
    
    bullish_ratio = ticker_data['bullish'] / total
    bearish_ratio = ticker_data['bearish'] / total
    
    # Score: bullish - bearish, normalized
    score = bullish_ratio - bearish_ratio
    
    # Weight by engagement (upvotes + comments)
    engagement = ticker_data['total_upvotes'] + ticker_data['total_comments']
    engagement_weight = min(engagement / 1000, 1.0)  # Cap at 1.0
    
    return score * (0.7 + 0.3 * engagement_weight)

def create_heatmap_data(ticker_sentiment, top_n=10):
    """Create data structure for heatmap visualization"""
    # Calculate scores and sort
    ticker_scores = []
    for ticker, data in ticker_sentiment.items():
        score = calculate_sentiment_score(data)
        ticker_scores.append({
            'ticker': ticker,
            'score': score,
            'bullish': data['bullish'],
            'bearish': data['bearish'],
            'neutral': data['neutral'],
            'total_mentions': data['total_mentions'],
            'total_upvotes': data['total_upvotes'],
            'total_comments': data['total_comments'],
            'subreddits': len(data['subreddits']),
            'subreddit_list': list(data['subreddits'])
        })
    
    # Sort by score (absolute value for most movement)
    ticker_scores.sort(key=lambda x: abs(x['score']) * x['total_mentions'], reverse=True)
    
    return ticker_scores[:top_n]

def display_heatmap(ticker_scores):
    """Display sentiment heatmap in terminal"""
    print("="*80)
    print("🔥 REDDIT SENTIMENT HEATMAP - TOP 10 SENTIMENT MOVERS")
    print("="*80)
    print()
    
    for i, ticker_data in enumerate(ticker_scores, 1):
        ticker = ticker_data['ticker']
        score = ticker_data['score']
        total = ticker_data['total_mentions']
        bullish = ticker_data['bullish']
        bearish = ticker_data['bearish']
        neutral = ticker_data['neutral']
        
        # Determine sentiment label
        if score > 0.3:
            sentiment_label = "🟢 STRONGLY BULLISH"
            bar_color = "█" * int(abs(score) * 50)
        elif score > 0.1:
            sentiment_label = "🟡 BULLISH"
            bar_color = "█" * int(abs(score) * 30)
        elif score < -0.3:
            sentiment_label = "🔴 STRONGLY BEARISH"
            bar_color = "█" * int(abs(score) * 50)
        elif score < -0.1:
            sentiment_label = "🟠 BEARISH"
            bar_color = "█" * int(abs(score) * 30)
        else:
            sentiment_label = "⚪ NEUTRAL"
            bar_color = "█" * 10
        
        print(f"{i}. ${ticker}")
        print(f"   {sentiment_label} (Score: {score:+.3f})")
        print(f"   📊 Mentions: {total} | 🟢 Bullish: {bullish} | 🔴 Bearish: {bearish} | ⚪ Neutral: {neutral}")
        print(f"   📈 Engagement: {ticker_data['total_upvotes']:,} upvotes, {ticker_data['total_comments']:,} comments")
        print(f"   📍 Subreddits: {', '.join(ticker_data['subreddit_list'][:5])}{'...' if len(ticker_data['subreddit_list']) > 5 else ''}")
        print(f"   [{bar_color}]")
        print()

def main():
    """Main function to generate sentiment heatmap"""
    print("="*80)
    print("🚀 XFINANCE REDDIT SENTIMENT HEATMAP GENERATOR")
    print("="*80)
    print(f"\n📡 Monitoring {len(SUBREDDITS)} subreddits...\n")
    
    # Fetch posts from all subreddits
    all_posts = []
    for subreddit in SUBREDDITS:
        posts = fetch_posts_from_subreddit(subreddit, limit=15)
        all_posts.extend(posts)
        print(f"  ✅ Found {len(posts)} posts with tickers")
    
    if not all_posts:
        print("\n❌ No posts with tickers found. Try again later.")
        return
    
    print(f"\n📊 Total posts collected: {len(all_posts)}")
    
    # Aggregate sentiment
    ticker_sentiment = aggregate_sentiment(all_posts)
    
    if not ticker_sentiment:
        print("❌ No tickers found in posts.")
        return
    
    # Create heatmap data
    heatmap_data = create_heatmap_data(ticker_sentiment, top_n=10)
    
    # Display heatmap
    display_heatmap(heatmap_data)
    
    # Save to JSON for potential dashboard use
    output_file = "sentiment_heatmap.json"
    with open(output_file, 'w') as f:
        json.dump(heatmap_data, f, indent=2, default=str)
    print(f"💾 Data saved to {output_file}")

# ============================================================================
# APP-INTEGRATION FUNCTIONS - Use these in your app
# ============================================================================

def get_stock_sentiment(ticker_symbol, limit_per_subreddit=20):
    """
    Get Reddit sentiment for a specific stock ticker.
    
    Args:
        ticker_symbol: Stock ticker (e.g., "AAPL", "NVDA")
        limit_per_subreddit: Number of posts to fetch per subreddit
    
    Returns:
        dict: Sentiment data for the stock
    """
    ticker_symbol = ticker_symbol.upper()
    
    # Fetch posts mentioning this ticker
    all_posts = []
    for subreddit in SUBREDDITS:
        posts = fetch_posts_from_subreddit(subreddit, limit=limit_per_subreddit, filter_spam=True, verbose=False)
        # Filter posts that mention this specific ticker
        for post in posts:
            if ticker_symbol in post['tickers']:
                all_posts.append(post)
    
    if not all_posts:
        return {
            'ticker': ticker_symbol,
            'sentiment_score': 0.0,
            'sentiment_label': 'NO_DATA',
            'total_mentions': 0,
            'bullish': 0,
            'bearish': 0,
            'neutral': 0,
            'total_upvotes': 0,
            'total_comments': 0,
            'subreddits': [],
            'message': f'No recent mentions found for {ticker_symbol} on Reddit'
        }
    
    # Aggregate sentiment
    ticker_sentiment = aggregate_sentiment(all_posts, verbose=False)
    
    if ticker_symbol not in ticker_sentiment:
        return {
            'ticker': ticker_symbol,
            'sentiment_score': 0.0,
            'sentiment_label': 'NO_DATA',
            'total_mentions': 0,
            'bullish': 0,
            'bearish': 0,
            'neutral': 0,
            'total_upvotes': 0,
            'total_comments': 0,
            'subreddits': [],
            'message': f'No sentiment data found for {ticker_symbol}'
        }
    
    ticker_data = ticker_sentiment[ticker_symbol]
    score = calculate_sentiment_score(ticker_data)
    
    # Determine sentiment label
    if score > 0.3:
        sentiment_label = "STRONGLY_BULLISH"
    elif score > 0.1:
        sentiment_label = "BULLISH"
    elif score < -0.3:
        sentiment_label = "STRONGLY_BEARISH"
    elif score < -0.1:
        sentiment_label = "BEARISH"
    else:
        sentiment_label = "NEUTRAL"
    
    return {
        'ticker': ticker_symbol,
        'sentiment_score': round(score, 3),
        'sentiment_label': sentiment_label,
        'total_mentions': ticker_data['total_mentions'],
        'bullish': ticker_data['bullish'],
        'bearish': ticker_data['bearish'],
        'neutral': ticker_data['neutral'],
        'total_upvotes': ticker_data['total_upvotes'],
        'total_comments': ticker_data['total_comments'],
        'subreddits': list(ticker_data['subreddits']),
        'recent_posts': [
            {
                'title': post['title'],
                'subreddit': post['subreddit'],
                'upvotes': post['upvotes'],
                'comments': post['comments'],
                'url': post['url'],
                'created': post['created'].isoformat() if isinstance(post['created'], datetime) else str(post['created'])
            }
            for post in ticker_data['posts'][:5]  # Top 5 recent posts
        ]
    }

def get_multiple_stocks_sentiment(ticker_symbols, limit_per_subreddit=15):
    """
    Get Reddit sentiment for multiple stock tickers at once.
    
    Args:
        ticker_symbols: List of stock tickers (e.g., ["AAPL", "NVDA", "TSLA"])
        limit_per_subreddit: Number of posts to fetch per subreddit
    
    Returns:
        dict: Sentiment data for all stocks
    """
    ticker_symbols = [t.upper() for t in ticker_symbols]
    
    # Fetch all posts
    all_posts = []
    for subreddit in SUBREDDITS:
        posts = fetch_posts_from_subreddit(subreddit, limit=limit_per_subreddit, filter_spam=True, verbose=False)
        all_posts.extend(posts)
    
    # Filter posts that mention any of our target tickers
    relevant_posts = []
    for post in all_posts:
        for ticker in post['tickers']:
            if ticker in ticker_symbols:
                relevant_posts.append(post)
                break
    
    if not relevant_posts:
        return {
            'stocks': {
                ticker: {
                    'ticker': ticker,
                    'sentiment_score': 0.0,
                    'sentiment_label': 'NO_DATA',
                    'total_mentions': 0,
                    'message': 'No recent mentions found'
                }
                for ticker in ticker_symbols
            }
        }
    
    # Aggregate sentiment
    ticker_sentiment = aggregate_sentiment(relevant_posts, verbose=False)
    
    # Build response for each requested ticker
    results = {}
    for ticker in ticker_symbols:
        if ticker in ticker_sentiment:
            ticker_data = ticker_sentiment[ticker]
            score = calculate_sentiment_score(ticker_data)
            
            if score > 0.3:
                sentiment_label = "STRONGLY_BULLISH"
            elif score > 0.1:
                sentiment_label = "BULLISH"
            elif score < -0.3:
                sentiment_label = "STRONGLY_BEARISH"
            elif score < -0.1:
                sentiment_label = "BEARISH"
            else:
                sentiment_label = "NEUTRAL"
            
            results[ticker] = {
                'ticker': ticker,
                'sentiment_score': round(score, 3),
                'sentiment_label': sentiment_label,
                'total_mentions': ticker_data['total_mentions'],
                'bullish': ticker_data['bullish'],
                'bearish': ticker_data['bearish'],
                'neutral': ticker_data['neutral'],
                'total_upvotes': ticker_data['total_upvotes'],
                'total_comments': ticker_data['total_comments'],
                'subreddits': list(ticker_data['subreddits'])
            }
        else:
            results[ticker] = {
                'ticker': ticker,
                'sentiment_score': 0.0,
                'sentiment_label': 'NO_DATA',
                'total_mentions': 0,
                'bullish': 0,
                'bearish': 0,
                'neutral': 0,
                'message': 'No recent mentions found'
            }
    
    return {'stocks': results}

def get_reddit_sentiment_summary(ticker_symbol):
    """
    Get a human-readable summary of Reddit sentiment for a stock.
    Perfect for displaying in an app.
    
    Args:
        ticker_symbol: Stock ticker (e.g., "AAPL")
    
    Returns:
        str: Human-readable sentiment summary
    """
    data = get_stock_sentiment(ticker_symbol)
    
    if data['total_mentions'] == 0:
        return f"No recent Reddit discussions found for ${ticker_symbol}."
    
    sentiment_desc = {
        'STRONGLY_BULLISH': 'very bullish',
        'BULLISH': 'bullish',
        'NEUTRAL': 'neutral',
        'BEARISH': 'bearish',
        'STRONGLY_BEARISH': 'very bearish',
        'NO_DATA': 'no data available'
    }
    
    sentiment = sentiment_desc.get(data['sentiment_label'], 'mixed')
    
    summary = f"Reddit sentiment for ${ticker_symbol} is {sentiment} "
    summary += f"(score: {data['sentiment_score']:+.2f}). "
    summary += f"Found {data['total_mentions']} mentions across {len(data['subreddits'])} subreddits. "
    summary += f"Breakdown: {data['bullish']} bullish, {data['bearish']} bearish, {data['neutral']} neutral. "
    summary += f"Total engagement: {data['total_upvotes']:,} upvotes, {data['total_comments']:,} comments."
    
    return summary

if __name__ == "__main__":
    main()


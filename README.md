newsflow.png
🎯 What's This All About?
Imagine having a personal news assistant that:

Watches news 24/7 so you don't have to

Answers questions instantly about anything happening right now

Learns continuously as news breaks

Summarizes what matters to you

That's NewsFlow AI! It's like ChatGPT but specifically for today's news that keeps getting smarter as news happens.

✨ What Makes This Special?
🔄 It's ALIVE! (Seriously)
Most AI systems use old data. Ours updates every 15 seconds with fresh news. Ask about a breaking story, and within 30 seconds of it being published, our AI knows about it!

🎯 Smart & Focused
"What's happening with AI today?" → Gets tech news

"Stock market updates?" → Focuses on business/finance

"Latest sports news?" → Filters to sports

...and it figures this out automatically!

🏠 Your Choice, Your Privacy
Use local AI (free, private) or cloud AI (powerful, fast) - you decide!

📊 See What's Happening
Beautiful dashboard shows:

How fast answers come

What news sources are being used

How accurate responses are

Live system performance

🚀 Get It Running in 5 Minutes!
No technical wizardry needed! Just follow these simple steps:

Step 1: Open Your Terminal/Command Prompt
Windows: Press Win + R, type cmd, press Enter

Mac: Press Cmd + Space, type Terminal, press Enter

Linux: Press Ctrl + Alt + T

Step 2: Copy & Paste These Commands
Type or copy each line below, press Enter after each:

bash
# 1. Download the project
git clone https://github.com/cornerstone-team/newsflow-ai.git

# 2. Go into the project folder
cd newsflow-ai

# 3. Create a safe space (virtual environment)
python -m venv venv

# 4. Activate it (do this EVERY TIME you start)
# For Windows:
venv\Scripts\activate
# For Mac/Linux:
source venv/bin/activate

# 5. Install everything needed
pip install -r requirements.txt
Step 3: Get Your Free NewsAPI Key
Go to newsapi.org

Click "Get API Key" (free tier is enough!)

Register with your email

Copy the key they give you

Step 4: Setup Your Config File
bash
# 1. Copy the example file
cp .env.example .env

# 2. Open it in a text editor
# On Windows:
notepad .env
# On Mac:
open -e .env
# On Linux:
nano .env
Edit the file to look like this:

env
NEWSAPI_KEY=paste_your_key_here
LLM_BACKEND=cloud  # Start with cloud (easier)
Save and close the file.

Step 5: Start It Up! 🎉
Open TWO terminal windows:

Terminal 1 - The Brain
bash
cd newsflow-ai
venv\Scripts\activate  # On Windows
# OR
source venv/bin/activate  # On Mac/Linux

python simple_news_rag.py
You'll see: "Pathway pipeline started..." - leave this running!

Terminal 2 - The Interface
bash
cd newsflow-ai
venv\Scripts\activate  # On Windows
# OR
source venv/bin/activate  # On Mac/Linux

streamlit run app.py
Step 6: Open Your Browser!
Look in Terminal 2 for a link like: http://localhost:8501

Click it or copy-paste into Chrome/Firefox

🎉 Welcome to NewsFlow AI!

🎮 How to Use It (It's Easy!)
First Things to Try:
Ask about today's news: "What's happening in the world?"

Get tech updates: "Latest technology news?"

Check business: "Stock market updates?"

Sports fan?: "What happened in sports today?"

See the Magic:
Watch the dashboard update with stats

Notice response times (usually 1-2 seconds!)

Check sources - see where info came from

Pro Tips:
Add "latest" or "today" for freshest news

Be specific: "AI news from today" vs just "news"

Try different categories: tech, business, politics, sports, entertainment, health

Watch the real-time stats grow as you use it

🎯 What Can You Ask?
Here are some fun examples to try:

text
"What's the biggest news story right now?"
"Tell me about technology company earnings"
"What happened in the football match yesterday?"
"Is there any news about climate change?"
"What are people saying about the new iPhone?"
"Any breaking news in politics?"
"Show me health news from today"
🛠️ Customize Your Experience
Want to Use Local AI? (No Internet Needed After Setup)
Install Ollama: ollama.ai (one-click install)

Pull a model:

bash
ollama pull mistral  # Fast & good
# OR
ollama pull llama2   # Popular choice
Edit your .env file:

env
LLM_BACKEND=local
Want Better Performance?
Edit .env and try:

env
POLL_INTERVAL=30  # Check news every 30 seconds (less load)
TOP_K=10         # Show 10 articles instead of 15
📱 See It In Action
The dashboard shows you:

✅ Today's Queries: How many questions you've asked

✅ Avg Response Time: How fast answers come

✅ Success Rate: How often we find good answers

✅ Top Categories: What you're interested in

✅ Popular Sources: Where news comes from

🐛 "Help! It's Not Working!"
Common Fixes:
Problem: "Module not found"
Solution: Run pip install -r requirements.txt again

Problem: "API key error"
Solution: Double-check your NewsAPI key in .env

Problem: "Connection refused"
Solution: Make sure you're in the newsflow-ai folder and venv is activated

Problem: "No news showing up"
Solution: Wait 1 minute, news comes every 15 seconds!

Still Stuck?
Check both terminals are running

Refresh your browser

Ask in Issues - we'll help!

🎓 Learn How It Works (For Curious Minds)
The Magic Behind:
NewsAPI feeds us live news

Pathway processes it instantly

AI understands and remembers

You ask questions and get smart answers

Why This Is Cool Tech:
Dynamic RAG: Most AI systems use static data. Ours updates live!

Real-time Processing: News → Answer in under 30 seconds

Smart Filtering: Knows what category you're asking about

Dual Brain: Local (private) or Cloud (powerful) - your choice

🚀 What's Next?
Soon you'll be able to:

Get email alerts for important news

Create personalized news feeds

Compare news from different sources

Add your own favorite news websites

Get summaries in different languages

👥 Meet the Team
We're Team Corner Stone - four friends who built this because we were tired of missing important news!

Arpit: The architect who made everything work together

Sagar: The AI wizard who made it smart

Keertan: The designer who made it beautiful

Asim: The data guy who keeps the news flowing

🙌 Want to Help Improve It?
We'd love your help! You can:

Report bugs (even tiny ones!)

Suggest features (what would help you?)

Share with friends who love tech

Star the repo ⭐ (helps others find it!)

📞 Need Help? Got Ideas?
GitHub Issues: Tell us what's up

Email: [Your feedback email]

Share your experience: We read every message!

<div align="center"> <h3>🎉 Ready to Try It?</h3>
bash
# Just copy-paste this to start:
git clone https://github.com/cornerstone-team/newsflow-ai.git
cd newsflow-ai
# Follow the steps above!
<p><i>"Finally, an AI that knows what happened this morning!"</i> - Early Tester</p> <p>Built with ❤️ for everyone who wants to stay informed without the overwhelm</p> </div>
💝 Support This Project
If this helps you stay informed:

⭐ Star the repo (top right!)

🐛 Report issues (help us improve)

📢 Share with friends (spread the knowledge)

💡 Suggest improvements (what would you add?)

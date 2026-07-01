"""
dashboard.py
------------
The Executive Intelligence Dashboard (Streamlit app).

This file is being built up in layers, section by section, matching the
7 required dashboard sections from the project brief:
  1. Company Overview        <- this layer
  2. Market Intelligence
  3. Opportunity Monitor
  4. Risk Monitor
  5. Sentiment Analysis
  6. Strategic Recommendations
  7. CEO Briefing

Run this with: streamlit run dashboard.py
(NOT with "python dashboard.py" - Streamlit apps need the special
"streamlit run" command, which starts a local web server and opens
your browser automatically.)
"""

import streamlit as st
import chromadb
from datetime import datetime

 
# CONFIG
 
CHROMA_PATH = "data/processed/chroma_db"
COLLECTION_NAME = "sap_intelligence"
COMPANY_NAME = "SAP"
INDUSTRY = "Enterprise Software / ERP"

# st.set_page_config must be the FIRST Streamlit command in the script.
# It sets the browser tab title and uses a wide layout (better for a
# dashboard with multiple columns/sections than the default narrow one).
st.set_page_config(
    page_title=f"{COMPANY_NAME} Strategic Intelligence Dashboard",
    layout="wide",
)


 
# DATA LOADING
#
# @st.cache_resource tells Streamlit: "run this function once, then reuse the result on every page refresh, instead of reconnecting to
# ChromaDB every single time the user interacts with the dashboard."
# This matters because Streamlit re-runs your ENTIRE script top-to-bottom on every interaction (every click, every widget change) - without
# caching, it would reconnect to the database constantly, which is wasteful.
 

@st.cache_resource
def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(COLLECTION_NAME)


def get_source_counts(collection):
    """
    Counts how many documents came from each source type (news, company,
    hackernews). ChromaDB doesn't have a built-in "count by category"
    query, so it fetches all metadata and counts manually in Python - this
    is fine at this scale (a few hundred documents).
    """
    all_data = collection.get()  # fetches everything (ids, metadatas, documents)
    source_counts = {}

    for metadata in all_data["metadatas"]:
        source = metadata.get("source_type", "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1

    return source_counts


 
# SECTION 1: COMPANY OVERVIEW
 

def render_company_overview(collection):
    st.title(f"{COMPANY_NAME} Strategic Intelligence Dashboard")
    st.caption("AI CEO: Strategic Intelligence Agent")

    source_counts = get_source_counts(collection)
    total_documents = sum(source_counts.values())
    num_sources = len(source_counts)

    # st.columns lets us lay out several metric boxes side by side
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Company", COMPANY_NAME)
    with col2:
        st.metric("Industry", INDUSTRY)
    with col3:
        st.metric("Documents collected", total_documents)
    with col4:
        st.metric("Data sources", num_sources)
    with col5:
        st.metric("Last updated", datetime.now().strftime("%Y-%m-%d %H:%M"))

    # A small breakdown of where the documents came from
    st.caption(
        "Source breakdown: "
        + ", ".join(f"{source}: {count}" for source, count in source_counts.items())
    )

    st.divider()


 
# SECTION 2: MARKET INTELLIGENCE
#
# This section shows recent items pulled directly from ChromaDB using
# semantic search - no LLM reasoning involved here, just fast retrieval
# and display. We run a few targeted queries (recent news, competitors,
# emerging tech, announcements) and show the top results for each.
 

@st.cache_resource
def get_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("BAAI/bge-small-en-v1.5")


def search_documents(collection, model, query, n_results=5):
    """Runs a semantic search query against ChromaDB and returns results."""
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)

    documents = []
    for doc_text, metadata in zip(results["documents"][0], results["metadatas"][0]):
        documents.append({
            "text": doc_text,
            "source_type": metadata.get("source_type", ""),
            "title": metadata.get("title", ""),
            "link": metadata.get("link", ""),
        })
    return documents


def render_document_list(documents):
    """Displays a list of documents as simple cards with a title, source
    tag, and clickable link."""
    for doc in documents:
        title = doc["title"] if doc["title"] else doc["text"][:80]
        link = doc["link"]
        source = doc["source_type"]

        if link:
            st.markdown(f"**[{title}]({link})**  \n`{source}`")
        else:
            st.markdown(f"**{title}**  \n`{source}`")


def render_market_intelligence(collection, model):
    st.header("Market Intelligence")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Recent news", "Competitor activity", "Emerging technologies", "Company announcements"
    ])

    with tab1:
        docs = search_documents(collection, model, f"recent news about {COMPANY_NAME}")
        render_document_list(docs)

    with tab2:
        docs = search_documents(collection, model, "Oracle Workday Microsoft Dynamics competitor activity")
        render_document_list(docs)

    with tab3:
        docs = search_documents(collection, model, "emerging technology trends artificial intelligence")
        render_document_list(docs)

    with tab4:
        docs = search_documents(collection, model, f"{COMPANY_NAME} official announcement press release")
        render_document_list(docs)

    st.divider()




 
# SECTIONS 3 + 4: OPPORTUNITY MONITOR + RISK MONITOR
#
# These sections call the AI CEO Agent (ai_ceo_agent.py) to get LLM-
# generated, evidence-backed opportunities and risks - this is where
# real reasoning happens, not just retrieval.
#
# @st.cache_data(ttl=600) caches the result for 10 minutes (600 seconds).
# This means: the LLM is only actually called once every 10 minutes,
# no matter how many times you click around the dashboard in between.
# This keeps the "live generation" behavior you asked for, while still
# avoiding a slow 10-30 second LLM call on every single click.
 

@st.cache_data(ttl=600)
def get_opportunities():
    import ai_ceo_agent
    return ai_ceo_agent.generate_monitor_items(
        "opportunity", f"opportunities for {COMPANY_NAME} growth and innovation"
    )


@st.cache_data(ttl=600)
def get_risks():
    import ai_ceo_agent
    return ai_ceo_agent.generate_monitor_items(
        "risk", f"risks and threats facing {COMPANY_NAME}"
    )


def render_monitor_items(items, level_field):
    """
    Shared rendering logic for both opportunities and risks - displays
    each item as a small card with title, level badge, evidence, and a
    confidence score progress bar.
    """
    if not items:
        st.warning("Could not generate items right now. Is Ollama running?")
        return

    for item in items:
        level = item.get(level_field, "Unknown")
        confidence = item.get("confidence_score", 0)

        # Color-code the level so High/Medium/Low are visually distinct
        if level == "High":
            level_display = f":red[{level}]"
        elif level == "Medium":
            level_display = f":orange[{level}]"
        else:
            level_display = f":green[{level}]"

        with st.container(border=True):
            st.markdown(f"**{item.get('title', 'Untitled')}**")
            st.markdown(f"Level: {level_display}")
            st.caption(item.get("evidence", ""))
            st.progress(confidence / 100, text=f"Confidence: {confidence}%")


def render_opportunity_monitor():
    st.header("Opportunity Monitor")
    with st.spinner("Analyzing opportunities (this calls the local LLM, may take 10-30s)..."):
        items = get_opportunities()
    render_monitor_items(items, "impact_level")
    st.divider()


def render_risk_monitor():
    st.header("Risk Monitor")
    with st.spinner("Analyzing risks (this calls the local LLM, may take 10-30s)..."):
        items = get_risks()
    render_monitor_items(items, "severity_level")
    st.divider()


 
# SECTION 5: SENTIMENT ANALYSIS
#
# Splits sentiment by source type: "news" source approximates news
# sentiment, "hackernews" source approximates public/community
# sentiment (the closest proxy to genuine public discussion).
# Uses the lexicon-based scoring from sentiment.py - fast, no extra
# model needed, and fully explainable.
 

def render_sentiment_analysis(collection):
    import sentiment
    import pandas as pd

    st.header("Sentiment Analysis")

    all_data = collection.get()
    documents_by_source = {"news": [], "hackernews": [], "company": []}

    for doc_text, metadata in zip(all_data["documents"], all_data["metadatas"]):
        source = metadata.get("source_type", "")
        if source in documents_by_source:
            documents_by_source[source].append({"text": doc_text})

    news_result = sentiment.analyze_documents(documents_by_source["news"])
    public_result = sentiment.analyze_documents(documents_by_source["hackernews"])

    col1, col2 = st.columns(2)
    with col1:
        st.metric("News sentiment", news_result["overall_label"],
                   delta=f"{news_result['average_score']:+.2f} avg score")
    with col2:
        st.metric("Public sentiment (Hacker News)", public_result["overall_label"],
                   delta=f"{public_result['average_score']:+.2f} avg score")

    # Build a small table for the bar chart: rows = Positive/Neutral/Negative,
    # columns = News / Public. st.bar_chart expects a dict-of-lists or
    # DataFrame-like structure where each key becomes a column.
    chart_data = {
        "News": [news_result["positive_count"], news_result["neutral_count"], news_result["negative_count"]],
        "Public (HN)": [public_result["positive_count"], public_result["neutral_count"], public_result["negative_count"]],
    }
    chart_df = pd.DataFrame(chart_data, index=["Positive", "Neutral", "Negative"])

    st.caption("Sentiment trend: document counts by sentiment category")
    st.bar_chart(chart_df)

    st.divider()


 
# SECTION 6: STRATEGIC RECOMMENDATIONS
#
# Calls ai_ceo_agent.generate_recommendation() for a few different
# strategic questions, displaying each as a full recommendation card
# with all the fields the brief asks for: recommendation, priority,
# supporting evidence, expected impact, risk level.
 

# A fixed set of strategic questions that are always asked. These cover the
# main angles your brief calls out (growth, technology, partnerships).
RECOMMENDATION_QUESTIONS = [
    f"What strategic action should {COMPANY_NAME} prioritize to drive growth?",
    f"What technology investment should {COMPANY_NAME} make next?",
    f"What partnership or market opportunity should {COMPANY_NAME} pursue?",
]


@st.cache_data(ttl=600)
def get_recommendations():
    import ai_ceo_agent
    results = []
    for question in RECOMMENDATION_QUESTIONS:
        result = ai_ceo_agent.generate_recommendation(question)
        if result is not None:
            results.append(result)
    return results


def render_recommendations():
    st.header("Strategic Recommendations")

    with st.spinner("Generating recommendations (calls the local LLM, may take a minute)..."):
        recommendations = get_recommendations()

    if not recommendations:
        st.warning("Could not generate recommendations right now.")
        return

    for rec in recommendations:
        priority = rec.get("priority", "Unknown")
        risk = rec.get("risk_level", "Unknown")

        priority_display = {"High": ":red[High]", "Medium": ":orange[Medium]", "Low": ":green[Low]"}.get(priority, priority)
        risk_display = {"High": ":red[High]", "Medium": ":orange[Medium]", "Low": ":green[Low]"}.get(risk, risk)

        with st.container(border=True):
            st.markdown(f"**{rec.get('recommendation', 'No recommendation')}**")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"Priority: {priority_display}")
            with col2:
                st.markdown(f"Risk level: {risk_display}")

            st.caption(f"Expected impact: {rec.get('expected_impact', '')}")

            with st.expander("Supporting evidence"):
                for item in rec.get("supporting_evidence", []):
                    st.markdown(f"- {item}")

    st.divider()


 
# SECTION 7: CEO BRIEFING
#
# The capstone section: a narrative executive summary answering
# "What happened? Why does it matter? What should management do next?"
 

@st.cache_data(ttl=600)
def get_ceo_briefing():
    import ai_ceo_agent
    return ai_ceo_agent.generate_ceo_briefing()


def render_ceo_briefing():
    st.header("CEO Briefing")
    st.caption(f'"If you were the CEO of {COMPANY_NAME} today, what would you do next and why?"')

    with st.spinner("Preparing executive briefing (calls the local LLM, may take a minute)..."):
        briefing = get_ceo_briefing()

    if briefing is None:
        st.warning("Could not generate the CEO briefing right now.")
        return

    with st.container(border=True):
        st.markdown("**What happened?**")
        st.write(briefing.get("what_happened", ""))

        st.markdown("**Why does it matter?**")
        st.write(briefing.get("why_it_matters", ""))

        st.markdown("**What should management do next?**")
        st.write(briefing.get("what_to_do_next", ""))

    st.divider()


 
# SECTION 8: AI AGENT WORKFLOW
#
# Unlike the other sections (which call ai_ceo_agent.py functions
# directly for a single retrieve -> generate step), this section calls
# agent.py's run_agent(), which executes the full multi-step workflow:
#
#     Goal -> Plan -> Retrieve -> Analyze -> Decide -> Recommend -> Validate
#
# Every stage's output is rendered explicitly, so the agent's reasoning
# process is visible, not just its final answer. This directly
# demonstrates: planning before execution, tool usage beyond the LLM,
# autonomous decision-making among named alternatives, and validation
# before presenting a result.
 

DEFAULT_AGENT_GOAL = f"Identify the most urgent strategic action {COMPANY_NAME} should take this quarter"


@st.cache_data(ttl=600)
def get_agent_run(goal):
    # The cache key includes `goal` itself (it's a function argument),
    # so a different query from the user produces a fresh agent run
    # instead of reusing a cached result for a different question.
    import agent
    return agent.run_agent(goal)


def render_agent_workflow():
    st.header("AI Agent Workflow")
    st.caption("Enter a strategic question for the AI CEO Agent to investigate, "
               "or leave blank to use the default goal.")

    # The user's own query becomes the agent's GOAL. This is what makes
    # the agent receive a genuine user request (per the project's agent
    # requirements) rather than always running one hardcoded goal.
    user_query = st.text_input(
        "Your question for the AI CEO Agent",
        placeholder=DEFAULT_AGENT_GOAL,
        key="agent_goal_input",
    )

    # If the user left the field blank, fall back to the default goal
    # rather than sending an empty string as the agent's objective.
    goal = user_query.strip() if user_query.strip() else DEFAULT_AGENT_GOAL

    run_button = st.button("Run Agent")

    # Only actually run the (slow, multi-LLM-call) agent when the user
    # explicitly clicks the button - not on every keystroke, which
    # Streamlit's re-run model would otherwise trigger.
    if not run_button and "last_agent_goal" not in st.session_state:
        st.info("Enter a question above and click \"Run Agent\" to start.")
        return

    if run_button:
        st.session_state["last_agent_goal"] = goal

    active_goal = st.session_state.get("last_agent_goal", goal)
    st.caption(f'Goal: "{active_goal}"')

    with st.spinner("Running full agent workflow - plan, retrieve, analyze, "
                     "decide, recommend, validate (multiple LLM calls, may take a few minutes)..."):
        result = get_agent_run(active_goal)

    # STEP: PLAN
    with st.expander("1. Plan", expanded=True):
        plan = result.get("plan", {})
        st.markdown("**Sub-questions the agent chose to investigate:**")
        for q in plan.get("sub_questions", []):
            st.markdown(f"- {q}")
        st.markdown(f"**Tools selected:** {', '.join(plan.get('tools_to_use', [])) or 'none'}")

    # STEP: RETRIEVE + TOOL USE
    with st.expander(f"2. Retrieve & Tool Use ({len(result.get('documents', []))} documents retrieved)"):
        for tool_result in result.get("tool_results", []):
            st.markdown(f"**{tool_result.get('tool')}**")
            st.json(tool_result)

    # STEP: ANALYZE
    with st.expander("3. Analyze"):
        analysis = result.get("analysis", {})
        st.markdown("**Key findings:**")
        for finding in analysis.get("key_findings", []):
            st.markdown(f"- {finding}")
        st.markdown(f"**Overall assessment:** {analysis.get('overall_assessment', '')}")

    # STEP: DECIDE
    with st.expander("4. Decide", expanded=True):
        decision = result.get("decision", {})
        decision_display = {
            "act_now": ":red[Act Now]",
            "monitor": ":orange[Monitor]",
            "investigate_further": ":blue[Investigate Further]",
        }.get(decision.get("decision"), decision.get("decision"))
        st.markdown(f"**Decision:** {decision_display}")
        st.markdown(f"**Reasoning:** {decision.get('reasoning', '')}")
        st.markdown(f"**Why other options were rejected:** {decision.get('rejected_alternatives', '')}")

    # STEP: RECOMMEND + VALIDATE (or MONITOR BRIEF)
    # The rendering here MUST branch on path_taken, since the monitor
    # path sets result["recommendation"] = None - calling .get() on None
    # would crash. Each branch produces a structurally different result.
    path_taken = result.get("path_taken", "act_now")

    if path_taken == "monitor":
        with st.container(border=True):
            st.markdown("### 5. Monitoring Brief :orange[— No Action Yet]")
            brief = result.get("monitoring_brief", {})
            st.markdown("**What to watch:**")
            for item in brief.get("watch_items", []):
                st.markdown(f"- {item}")
            st.markdown(f"**Trigger condition for escalation:** {brief.get('trigger_condition', '')}")
            st.info("The agent decided the situation does not yet warrant a specific "
                    "recommendation. No validation step was run because no "
                    "recommendation was generated.")

    else:
        if path_taken == "investigate_further":
            with st.expander("5a. Follow-up Investigation"):
                st.markdown(f"**Agent's follow-up question:** {result.get('followup_question', '')}")
                st.caption(f"Evidence set expanded to {len(result.get('documents', []))} "
                           f"documents before proceeding to recommendation.")

        recommendation = result.get("recommendation") or {}
        validation = result.get("validation") or {}
        approved = validation.get("approved", False)
        retried = result.get("retried", False)

        with st.container(border=True):
            step_label = "5b. Final Recommendation" if path_taken == "investigate_further" else "5. Final Recommendation"
            status_badge = ":green[✓ Validated]" if approved else ":red[✗ Not Validated]"
            retry_badge = " :orange[(retried after initial rejection)]" if retried else ""
            st.markdown(f"### {step_label} — {status_badge}{retry_badge}")
            st.markdown(f"**{recommendation.get('recommendation', '')}**")
            st.caption(f"Priority: {recommendation.get('priority', '')} | "
                       f"Risk level: {recommendation.get('risk_level', '')}")
            st.markdown(f"**Validation notes:** {validation.get('validation_notes', '')}")

            if not approved:
                st.warning("This recommendation did not pass validation against the "
                           "evidence and should be treated with caution or re-investigated.")

            if retried:
                original_recommendation = result.get("original_recommendation") or {}
                original_validation = result.get("original_validation") or {}
                with st.expander("Why the agent retried: original (rejected) attempt"):
                    st.markdown(f"**Original recommendation:** {original_recommendation.get('recommendation', '')}")
                    st.markdown(f"**Rejection reason:** {original_validation.get('validation_notes', '')}")
                    st.caption("The agent used this rejection reason to generate a new "
                               "recommendation attempt, shown above, rather than retrying blindly.")

    st.divider()


def main():
    collection = get_chroma_collection()
    model = get_embedding_model()

    render_company_overview(collection)
    render_market_intelligence(collection, model)
    render_opportunity_monitor()
    render_risk_monitor()
    render_sentiment_analysis(collection)
    render_recommendations()
    render_ceo_briefing()
    render_agent_workflow()


if __name__ == "__main__":
    main()
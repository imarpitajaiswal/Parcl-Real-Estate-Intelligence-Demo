"""PARCL | Real Estate Buyer Intelligence
Full replacement Streamlit app — Steps 12–14.
Run from the project root: python -m streamlit run app.py

Requires outputs from the previous data preparation, modeling, and profiling steps.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
# Public demonstration: independently generated fictional data
DATA = ROOT / 'demo_data'
OUTPUT = ROOT / 'demo_data'

SEGMENT_COLORS = {
    'High-Volume Property Buyers': '#E3A647',
    'Mixed-Portfolio Buyers': '#4E7AC7',
    'Residential-Focused Buyers': '#139C94',
}
PAGES = [
    'Executive Overview', 'Buyer Segmentation', 'Investment Intelligence',
    'Geographic Analytics', 'ML Model Insights',
]

st.set_page_config(page_title='Parcl | Real Estate Intelligence',
                   page_icon='🏙️', layout='wide', initial_sidebar_state='expanded')
st.markdown('''<style>
/* Explicit light appearance for the main content, including dark-mode browsers */
[data-testid="stAppViewContainer"] {background: #f5f7fb; color: #172b4d;}
[data-testid="stAppViewContainer"] .block-container h1,
[data-testid="stAppViewContainer"] .block-container h2,
[data-testid="stAppViewContainer"] .block-container h3,
[data-testid="stAppViewContainer"] .block-container p,
[data-testid="stAppViewContainer"] .block-container label,
[data-testid="stAppViewContainer"] [data-testid="stMetricLabel"],
[data-testid="stAppViewContainer"] [data-testid="stMetricValue"],
[data-testid="stAppViewContainer"] [data-testid="stMetricDelta"],
[data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] {color: #172b4d !important;}
[data-testid="stAppViewContainer"] [data-testid="stMetric"] {color:#172b4d;}
[data-testid="stSidebar"] {background: #14243b;}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {color:#f1f5f9 !important;}
[data-testid="stMetric"] {background:white;border:1px solid #e1e7ef;
    border-radius:12px;padding:14px;}
[data-testid="stPlotlyChart"] {background:white;border-radius:12px;}
.block-container {padding-top:1.5rem;padding-bottom:3rem;}
</style>''', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data():
    buyers_path = DATA / 'buyer_segments.csv'
    properties_path = DATA / 'properties_cleaned.csv'
    if not buyers_path.exists() or not properties_path.exists():
        raise FileNotFoundError('Missing synthetic demo CSV files in demo_data/. Run python src/generate_demo_data.py first.')
    buyers = pd.read_csv(buyers_path)
    properties = pd.read_csv(properties_path)
    required = {
        'client_id', 'segment_name', 'cluster', 'country', 'region', 'client_type',
        'acquisition_purpose', 'loan_applied', 'referral_channel',
        'properties_purchased', 'total_investment_usd', 'avg_property_price_usd',
        'office_purchase_share', 'buyer_age', 'satisfaction_score',
        'loan_applied_binary', 'is_investment_buyer',
    }
    missing = required.difference(buyers.columns)
    if missing:
        raise ValueError(f'Missing buyer columns: {sorted(missing)}')
    if not buyers['client_id'].is_unique:
        raise ValueError('Duplicate client IDs in buyer_segments.csv')
    if not {'listing_status', 'client_ref', 'sale_price', 'unit_category', 'listing_id'}.issubset(properties.columns):
        raise ValueError('Missing required property columns')
    buyers['buyer_age'] = pd.to_numeric(buyers['buyer_age'], errors='coerce')
    buyers['total_investment_usd'] = pd.to_numeric(buyers['total_investment_usd'], errors='raise')
    properties['sale_price'] = pd.to_numeric(properties['sale_price'], errors='raise')
    return buyers, properties


@st.cache_data(show_spinner=False)
def load_model_results():
    # No model evaluation is available for simulated labels.
    return {
        'kmeans': pd.DataFrame(),
        'hierarchical': pd.DataFrame(),
        'optimization': pd.DataFrame(),
    }


@st.cache_data(show_spinner=False)
def load_model_summary():
    path = OUTPUT / 'segment_summary.json'
    if not path.exists():
        return {}
    with path.open(encoding='utf-8') as f:
        return json.load(f)



st.warning(
    "PUBLIC DEMONSTRATION — All buyer and property records "
    "are independently generated synthetic data. "
    "Segment labels are simulated for interface demonstration "
    "and are not predictions from the internship ML model."
)


def money(value):
    value = float(value)
    if abs(value) >= 1e9:
        return f'${value / 1e9:,.2f}B'
    if abs(value) >= 1e6:
        return f'${value / 1e6:,.2f}M'
    if abs(value) >= 1e3:
        return f'${value / 1e3:,.1f}K'
    return f'${value:,.2f}'


def show_chart(fig, height=420):
    fig.update_layout(template='plotly_white', height=height,
                      margin=dict(l=30, r=30, t=65, b=35),
                      font=dict(family='Arial', color='#172b4d', size=12),
                      paper_bgcolor='white', plot_bgcolor='white')
    st.plotly_chart(fig, width='stretch')


def header(title, description):
    st.title(title)
    st.caption(description)
    st.divider()


def sidebar_filters(buyers):
    st.sidebar.title('🏙️ PARCL')
    st.sidebar.caption('REAL ESTATE INTELLIGENCE')
    st.sidebar.divider()
    page = st.sidebar.radio('Navigation', PAGES)
    st.sidebar.divider()
    st.sidebar.subheader('Global Filters')

    selected = buyers.copy()
    countries = st.sidebar.multiselect('Country', sorted(buyers['country'].dropna().unique()), key='country_filter')
    if countries:
        selected = selected[selected['country'].isin(countries)]
    regions = st.sidebar.multiselect('Region', sorted(selected['region'].dropna().unique()), key='region_filter')
    if regions:
        selected = selected[selected['region'].isin(regions)]
    purposes = st.sidebar.multiselect('Acquisition Purpose', sorted(buyers['acquisition_purpose'].dropna().unique()), key='purpose_filter')
    if purposes:
        selected = selected[selected['acquisition_purpose'].isin(purposes)]
    types = st.sidebar.multiselect('Client Type', sorted(buyers['client_type'].dropna().unique()), key='client_type_filter')
    if types:
        selected = selected[selected['client_type'].isin(types)]

    st.sidebar.subheader('Advanced Filters')
    segments = st.sidebar.multiselect('Buyer Segment', sorted(buyers['segment_name'].dropna().unique()), key='segment_filter')
    if segments:
        selected = selected[selected['segment_name'].isin(segments)]
    loan = st.sidebar.selectbox('Loan Application', ['All', 'Yes', 'No'], key='loan_filter')
    if loan != 'All':
        selected = selected[selected['loan_applied'].eq(loan)]

    spending_max = max(1.0, float(np.ceil(buyers['total_investment_usd'].max() / 1e5) / 10))
    spend_range = st.sidebar.slider('Total Spending (USD Millions)',
                                    min_value=0.0, max_value=spending_max,
                                    value=(0.0, spending_max), step=0.1,
                                    key='spending_range')
    # Small rounding margin at the max ensures the unfiltered default includes every record.
    upper = float('inf') if spend_range[1] >= spending_max else spend_range[1] * 1e6
    selected = selected[selected['total_investment_usd'].between(spend_range[0] * 1e6, upper)]

    if st.sidebar.checkbox('Filter by Individual Buyer Age', value=False, key='enable_age'):
        ages = buyers.loc[buyers['client_type'].eq('Individual'), 'buyer_age'].dropna()
        if not ages.empty:
            lo, hi = int(np.floor(ages.min())), int(np.ceil(ages.max()))
            age_range = st.sidebar.slider('Buyer Age', min_value=lo, max_value=hi,
                                          value=(lo, hi), key='age_filter')
            selected = selected[
                selected['client_type'].eq('Individual') &
                selected['buyer_age'].between(age_range[0], age_range[1])
            ]
        else:
            selected = selected.iloc[0:0]

    st.sidebar.divider()
    st.sidebar.caption(f'Showing {len(selected):,} of {len(buyers):,} buyers')
    st.sidebar.caption('Unified Mentor × Parcl Co. Limited')
    return page, selected


def show_kpis(df):
    count = len(df)
    spending = df['total_investment_usd'].sum()
    purchases = df['properties_purchased'].sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Total Buyers', f'{count:,}')
    c2.metric('Property Spending', money(spending))
    c3.metric('Properties Purchased', f'{purchases:,.0f}')
    c4.metric('Avg. Buyer Spending', money(spending / count if count else 0))


def segment_summary(df):
    return (df.groupby('segment_name', as_index=False)
            .agg(buyers=('client_id', 'count'), purchases=('properties_purchased', 'sum'),
                 total_spending=('total_investment_usd', 'sum'),
                 avg_spending=('total_investment_usd', 'mean'),
                 avg_purchases=('properties_purchased', 'mean'),
                 avg_office_share=('office_purchase_share', 'mean'),
                 satisfaction=('satisfaction_score', 'mean')))


def spending_concentration_chart(df):
    st.subheader('Buyer Spending Concentration')
    spending = df['total_investment_usd'].dropna().sort_values(ascending=False).reset_index(drop=True)
    total = spending.sum()
    if spending.empty or total <= 0:
        st.info('No spending data available.')
        return
    concentration = pd.DataFrame({
        'buyer_percentile': np.arange(1, len(spending) + 1) / len(spending) * 100,
        'cumulative_spending_pct': spending.cumsum() / total * 100,
    })
    fig = px.area(concentration, x='buyer_percentile', y='cumulative_spending_pct',
                  title='Cumulative Spending by Top Buyers',
                  labels={'buyer_percentile': 'Top Buyers Included (%)',
                          'cumulative_spending_pct': 'Cumulative Spending Share (%)'})
    fig.update_traces(line_color='#3976D2', fillcolor='rgba(57,118,210,0.15)')
    fig.update_layout(yaxis_range=[0, 100], xaxis_range=[0, 100])
    show_chart(fig)
    n = max(1, int(np.ceil(len(spending) * .10)))
    st.metric('Spending Share of Top 10% Buyers', f'{spending.head(n).sum() / total * 100:.2f}%')
    st.caption('Calculated within the current filtered buyer population.')


def advanced_segment_comparison(df):
    st.subheader('Advanced Segment Comparison')
    summary = (df.groupby('segment_name', as_index=False)
               .agg(buyers=('client_id', 'count'),
                    average_spending=('total_investment_usd', 'mean'),
                    average_purchases=('properties_purchased', 'mean'),
                    office_share=('office_purchase_share', 'mean'),
                    average_satisfaction=('satisfaction_score', 'mean'),
                    loan_rate=('loan_applied_binary', 'mean')))
    summary[['office_share', 'loan_rate']] *= 100
    descriptions = {
        'average_spending': 'Average Spending (USD)',
        'average_purchases': 'Average Purchases',
        'office_share': 'Average Office Purchase Share (%)',
        'average_satisfaction': 'Average Satisfaction Score',
        'loan_rate': 'Loan Application Rate (%)',
    }
    metric = st.selectbox('Compare segments by', list(descriptions),
                          format_func=descriptions.get, key='comparison_metric')
    fig = px.bar(summary, x='segment_name', y=metric, color='segment_name',
                 color_discrete_map=SEGMENT_COLORS,
                 title=descriptions[metric],
                 labels={'segment_name': 'Buyer Segment', metric: descriptions[metric]})
    fig.update_layout(showlegend=False, xaxis_tickangle=-15)
    show_chart(fig)
    st.dataframe(summary.round(2), hide_index=True, width='stretch')


def downloadable_reports(df, key):
    st.subheader('Download Business Intelligence Reports')
    segments = segment_summary(df)
    countries = (df.groupby('country', as_index=False)
                 .agg(buyers=('client_id', 'count'),
                      total_spending_usd=('total_investment_usd', 'sum'),
                      properties_purchased=('properties_purchased', 'sum')))
    c1, c2 = st.columns(2)
    c1.download_button('Download Segment Performance CSV',
                       data=segments.to_csv(index=False).encode('utf-8'),
                       file_name='parcl_segment_performance.csv', mime='text/csv',
                       width='stretch', key=f'{key}_segment_download')
    c2.download_button('Download Geographic Summary CSV',
                       data=countries.to_csv(index=False).encode('utf-8'),
                       file_name='parcl_geographic_summary.csv', mime='text/csv',
                       width='stretch', key=f'{key}_country_download')
    st.caption('These are aggregated exports. Small segments can still be sensitive; confirm permissions before sharing.')


def executive_overview(df):
    header('Executive Overview', 'Real estate purchasing analytics and AI-derived buyer profiles')
    show_kpis(df)
    st.subheader('Customer Portfolio Overview')
    left, right = st.columns(2)
    with left:
        counts = df['segment_name'].value_counts().rename_axis('segment_name').reset_index(name='buyers')
        fig = px.pie(counts, names='segment_name', values='buyers', hole=.58,
                     color='segment_name', color_discrete_map=SEGMENT_COLORS,
                     title='Buyer Segment Distribution')
        show_chart(fig)
    with right:
        summary = segment_summary(df)
        fig = px.bar(summary, x='segment_name', y='total_spending', color='segment_name',
                     color_discrete_map=SEGMENT_COLORS, title='Property Spending by Segment',
                     labels={'segment_name': 'Buyer Segment', 'total_spending': 'Spending (USD)'})
        fig.update_layout(showlegend=False, xaxis_tickangle=-15)
        show_chart(fig)
    left, right = st.columns(2)
    with left:
        fig = px.pie(df, names='acquisition_purpose', hole=.5,
                     title='Acquisition Purpose', color_discrete_sequence=['#4E7AC7', '#139C94'])
        show_chart(fig)
    with right:
        channels = df['referral_channel'].value_counts().rename_axis('channel').reset_index(name='buyers')
        fig = px.bar(channels, x='channel', y='buyers', color='buyers',
                     color_continuous_scale='Blues', title='Acquisition Channels')
        show_chart(fig)
    st.subheader('Segment Performance')
    st.dataframe(segment_summary(df).round(2), hide_index=True, width='stretch')
    st.divider()
    spending_concentration_chart(df)
    st.divider()
    downloadable_reports(df, 'executive')


def buyer_segmentation(df):
    header('AI Buyer Segmentation', 'Observed purchasing patterns; clusters are descriptive, not financial-risk classifications')
    show_kpis(df)
    summary = segment_summary(df)
    left, right = st.columns(2)
    with left:
        fig = px.bar(summary, x='segment_name', y='buyers', color='segment_name',
                     color_discrete_map=SEGMENT_COLORS, title='Segment Population')
        fig.update_layout(showlegend=False, xaxis_tickangle=-15)
        show_chart(fig)
    with right:
        fig = px.bar(summary, x='segment_name', y='avg_spending', color='segment_name',
                     color_discrete_map=SEGMENT_COLORS, title='Average Buyer Spending')
        fig.update_layout(showlegend=False, xaxis_tickangle=-15)
        show_chart(fig)
    st.subheader('Buyer Cluster Profile')
    choice = st.selectbox('Choose Buyer Segment', sorted(df['segment_name'].unique()), key='profile_segment')
    subset = df[df['segment_name'].eq(choice)]
    c1, c2, c3 = st.columns(3)
    c1.metric('Buyers', f'{len(subset):,}')
    c2.metric('Avg. Purchases', f"{subset['properties_purchased'].mean():.2f}")
    c3.metric('Avg. Spending', money(subset['total_investment_usd'].mean()))
    st.dataframe(summary.round(2), hide_index=True, width='stretch')
    st.info('Segments do not establish income, net worth, verified investment intention, or causation.')
    st.divider()
    advanced_segment_comparison(df)
    st.divider()
    downloadable_reports(df, 'segmentation')


def investment_intelligence(df, properties):
    header('Investment Intelligence', 'Property expenditure, financing indicators, and purchasing preferences')
    show_kpis(df)
    left, right = st.columns(2)
    with left:
        fig = px.histogram(df, x='total_investment_usd', nbins=35, color='segment_name',
                           color_discrete_map=SEGMENT_COLORS,
                           title='Buyer Spending Distribution',
                           labels={'total_investment_usd': 'Total Property Spending (USD)'})
        show_chart(fig)
    with right:
        fig = px.box(df, x='segment_name', y='avg_property_price_usd', color='segment_name',
                     color_discrete_map=SEGMENT_COLORS, title='Average Purchased Property Price',
                     labels={'avg_property_price_usd': 'Average Property Price (USD)'})
        fig.update_layout(showlegend=False, xaxis_tickangle=-15)
        show_chart(fig)
    left, right = st.columns(2)
    with left:
        loan = df.groupby(['acquisition_purpose', 'loan_applied']).size().reset_index(name='buyers')
        fig = px.bar(loan, x='acquisition_purpose', y='buyers', color='loan_applied',
                     barmode='group', title='Loan Applications by Purpose')
        show_chart(fig)
    with right:
        fig = px.scatter(df, x='properties_purchased', y='total_investment_usd',
                         color='segment_name', color_discrete_map=SEGMENT_COLORS,
                         opacity=.65, title='Purchase Frequency vs Total Spending')
        show_chart(fig)
    # Scope property-level charts to the currently filtered buyers.
    sold = properties[properties['listing_status'].eq('Sold')]
    sold = sold[sold['client_ref'].isin(df['client_id'])]
    st.subheader('Purchased Property Categories (Filtered Buyers)')
    if sold.empty:
        st.info('No sold property records for selected buyers.')
    else:
        categories = sold.groupby('unit_category', as_index=False).agg(
            properties=('listing_id', 'count'), spending_usd=('sale_price', 'sum'))
        a, b = st.columns(2)
        with a:
            show_chart(px.bar(categories, x='unit_category', y='properties',
                              title='Number of Sold Properties'))
        with b:
            show_chart(px.bar(categories, x='unit_category', y='spending_usd',
                              title='Recorded Sales by Property Type'))
        if not np.isclose(sold['sale_price'].sum(), df['total_investment_usd'].sum(), atol=.01, rtol=0):
            st.warning('Financial totals do not reconcile for the current filter. Check source data.')
    st.caption('Loan applied means an application, not approval. Buyer spending includes home and investment purchases.')
    st.divider()
    spending_concentration_chart(df)


def geographic_analytics(df):
    header('Geographic Buyer Intelligence', 'Country and region refer to buyer residence, not property location')
    show_kpis(df)
    country = df.groupby('country', as_index=False).agg(
        buyers=('client_id', 'count'), total_spending=('total_investment_usd', 'sum'))
    fig = px.choropleth(country, locations='country', locationmode='country names',
                        color='buyers', hover_name='country', hover_data=['total_spending'],
                        color_continuous_scale='Blues', title='International Buyer Distribution')
    show_chart(fig, height=500)
    a, b = st.columns(2)
    with a:
        top = country.nlargest(10, 'buyers')
        show_chart(px.bar(top, x='country', y='buyers', color='buyers',
                          color_continuous_scale='Blues', title='Top Buyer Countries'))
    with b:
        region = df['region'].value_counts().head(10).rename_axis('region').reset_index(name='buyers')
        show_chart(px.bar(region, x='region', y='buyers', color='buyers',
                          color_continuous_scale='Teal', title='Top Buyer Regions'))
    st.subheader('Country Financial Summary')
    st.dataframe(country.sort_values('total_spending', ascending=False).round(2),
                 hide_index=True, width='stretch')
    st.caption('These figures reflect the residence recorded for each buyer.')
    st.divider()
    downloadable_reports(df, 'geographic')


def ml_model_insights(df, models, summary):
    header(
        "Segmentation Methodology | Synthetic Demo",
        "Demonstration of buyer segmentation concepts "
        "using independently generated fictional records."
    )

    st.info(
        "PUBLIC DEMO: Segment assignments are simulated. "
        "No clustering model was trained or evaluated "
        "on this demonstration dataset."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Buyers in Current Filter",
        f"{len(df):,}"
    )

    c2.metric(
        "Segments in Current Filter",
        str(df["segment_name"].nunique())
    )

    c3.metric(
        "Model Evaluation",
        "Not Applicable"
    )

    st.subheader("Demonstration Segment Distribution")

    counts = (
        df["segment_name"]
        .value_counts()
        .rename_axis("segment_name")
        .reset_index(name="buyers")
    )

    if not counts.empty:
        fig = px.bar(
            counts,
            x="segment_name",
            y="buyers",
            color="segment_name",
            color_discrete_map=SEGMENT_COLORS,
            title="Simulated Buyer Segment Distribution"
        )
        show_chart(fig)

    st.subheader("How the Original Research Approach Works")

    st.markdown("""
    The original internship project explores buyer segmentation
    through an unsupervised machine-learning workflow:

    1. Data validation and cleaning
    2. Buyer-level feature engineering
    3. Feature preprocessing and scaling
    4. K-Means and hierarchical clustering
    5. Cluster evaluation and diagnostic analysis
    6. Business-oriented segment profiling

    **This public demonstration is different.**

    It uses fictional buyer and property transactions,
    with segment labels assigned by the synthetic-data generator.

    The displayed charts illustrate how segmentation insights
    can be explored through an interactive dashboard.
    They do not constitute independently validated
    machine-learning findings.
    """)

    st.subheader("Interpretation Limitations")

    st.warning(
        "Simulated segment labels should not be used to infer "
        "customer wealth, financial risk, investment suitability, "
        "or real-world purchasing behavior."
    )

    st.caption(
        "Synthetic data demonstration. "
        "Original internship model evaluation results are "
        "intentionally excluded."
    )


def main():
    try:
        buyers, properties = load_data()
        models = load_model_results()
        summary = load_model_summary()
    except (FileNotFoundError, ValueError, OSError, pd.errors.ParserError) as exc:
        st.error(f'Unable to load project data: {exc}')
        st.stop()

    page, filtered = sidebar_filters(buyers)
    if filtered.empty:
        st.warning('No buyers match your filters. Please adjust your selections.')
        st.stop()
    if page == 'Executive Overview':
        executive_overview(filtered)
    elif page == 'Buyer Segmentation':
        buyer_segmentation(filtered)
    elif page == 'Investment Intelligence':
        investment_intelligence(filtered, properties)
    elif page == 'Geographic Analytics':
        geographic_analytics(filtered)
    elif page == 'ML Model Insights':
        ml_model_insights(filtered, models, summary)


if __name__ == '__main__':
    main()

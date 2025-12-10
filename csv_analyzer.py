import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# ----------------- Page Config -----------------
st.set_page_config(page_title="🧹 CSV Analyzer", layout="wide", page_icon="📊")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {background-color: #0E1117; color: #FAFAFA;}
    h1, h2, h3 {color: #FF4B4B;}
    .stDataFrame {background-color: #262730;}
    .small-chart div {height: 130px !important;}
    </style>
""", unsafe_allow_html=True)

# ----------------- Header -----------------
st.title("📊 CSV Analyzer & Cleaner")
st.markdown("Upload any CSV file to **preview, clean, filter, and visualize interactively**. Dashboard-style with KPIs and sparklines.")

# ----------------- File Upload -----------------
uploaded_file = st.file_uploader("📂 Upload a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, low_memory=False)

    # ----------------- Cleaning Form -----------------
    with st.form("cleaning_form"):
        st.sidebar.header("🧹 Data Cleaning Options")
        strip_cols = st.sidebar.checkbox("Strip whitespace from column names", value=True)
        remove_dupes = st.sidebar.checkbox("Remove duplicate rows")

        st.sidebar.subheader("⚠️ Null Value Handling")
        null_option = st.sidebar.selectbox(
            "Choose how to handle missing values",
            ("None", "Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode", "Custom Fill")
        )

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include='object').columns.tolist()

        # Filters
        st.sidebar.subheader("🔍 Filters")
        filter_dict = {}
        for col in numeric_cols:
            try:
                min_val, max_val = float(df[col].min()), float(df[col].max())
                selected_range = st.sidebar.slider(f"{col} range", min_val, max_val, (min_val, max_val))
                filter_dict[col] = selected_range
            except Exception:
                continue
        for col in categorical_cols:
            options = df[col].astype(str).unique().tolist()
            default_opts = options[:10] if len(options) > 10 else options
            selected_options = st.sidebar.multiselect(f"{col} values", options, default=default_opts)
            filter_dict[col] = selected_options

        # Sparkline control (choose a numeric column for KPI trend)
        st.sidebar.subheader("📈 KPI Sparkline")
        spark_num_col = st.sidebar.selectbox(
            "Numeric column for trend sparkline",
            numeric_cols if len(numeric_cols) > 0 else ["(none)"]
        )

        # Submit button
        submitted = st.form_submit_button("✅ Apply Cleaning & Show Dashboard")

    # ----------------- Apply Cleaning Only After Submit -----------------
    if submitted:
        if strip_cols:
            df.columns = df.columns.str.strip()
        if remove_dupes:
            df.drop_duplicates(inplace=True)

        if null_option == "Drop Rows":
            df.dropna(inplace=True)
        elif null_option == "Fill with Mean":
            for col in numeric_cols:
                df[col].fillna(df[col].mean(), inplace=True)
        elif null_option == "Fill with Median":
            for col in numeric_cols:
                df[col].fillna(df[col].median(), inplace=True)
        elif null_option == "Fill with Mode":
            for col in df.columns:
                mode_vals = df[col].mode()
                if len(mode_vals) > 0:
                    df[col].fillna(mode_vals[0], inplace=True)
        elif null_option == "Custom Fill":
            custom_value = st.sidebar.text_input("Enter custom fill value")
            if custom_value != "":
                df = df.fillna(custom_value)

        # Apply filters
        df_filtered = df.copy()
        for col, val in filter_dict.items():
            if col in numeric_cols and isinstance(val, tuple):
                df_filtered = df_filtered[(df_filtered[col] >= val[0]) & (df_filtered[col] <= val[1])]
            elif col in df_filtered.columns and not isinstance(val, tuple):
                df_filtered = df_filtered[df_filtered[col].astype(str).isin([str(v) for v in val])]

        # ----------------- KPI Summary Row -----------------
        st.markdown("### 📊 Dataset overview")
        rows = len(df_filtered)
        cols = len(df_filtered.columns)
        missing = int(df_filtered.isnull().sum().sum())
        mem_usage = df_filtered.memory_usage(deep=True).sum() / 1024**2  # MB

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(label="Total rows", value=f"{rows:,}")
        k2.metric(label="Total columns", value=f"{cols:,}")
        k3.metric(label="Missing values", value=f"{missing:,}")
        k4.metric(label="Memory usage (MB)", value=f"{mem_usage:.2f}")

        # ----------------- KPI Sparklines -----------------
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.caption("Trend sparkline")
            if spark_num_col in df_filtered.columns and rows > 0 and spark_num_col in numeric_cols:
                fig_trend = px.line(
                    df_filtered.reset_index(),
                    x="index", y=spark_num_col,
                    height=130, template="plotly_dark"
                )
                fig_trend.update_layout(margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
                fig_trend.update_xaxes(visible=False)
                fig_trend.update_yaxes(visible=False)
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.write("No numeric column selected.")

        with s2:
            st.caption("Missing per row")
            miss_per_row = df_filtered.isnull().sum(axis=1)
            fig_missing_row = px.line(
                miss_per_row.reset_index(),
                x="index", y=0, height=130, template="plotly_dark", color_discrete_sequence=["#FF4B4B"]
            )
            fig_missing_row.update_layout(margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
            fig_missing_row.update_xaxes(visible=False)
            fig_missing_row.update_yaxes(visible=False)
            st.plotly_chart(fig_missing_row, use_container_width=True)

        with s3:
            st.caption("Unique values per column")
            uniques = df_filtered.nunique().sort_values(ascending=False).head(15)
            fig_uniques = px.bar(
                uniques[::-1],
                height=130, template="plotly_dark", color_discrete_sequence=["#00CC96"]
            )
            fig_uniques.update_layout(margin=dict(l=10, r=10, t=20, b=10))
            fig_uniques.update_xaxes(visible=False)
            fig_uniques.update_yaxes(visible=False)
            st.plotly_chart(fig_uniques, use_container_width=True)

        with s4:
            st.caption("Std dev of numeric columns")
            if len(numeric_cols) > 0:
                stds = df_filtered[numeric_cols].std(numeric_only=True).sort_values(ascending=False).head(15)
                fig_stds = px.bar(
                    stds[::-1],
                    height=130, template="plotly_dark", color_discrete_sequence=["#AB63FA"]
                )
                fig_stds.update_layout(margin=dict(l=10, r=10, t=20, b=10))
                fig_stds.update_xaxes(visible=False)
                fig_stds.update_yaxes(visible=False)
                st.plotly_chart(fig_stds, use_container_width=True)
            else:
                st.write("No numeric columns.")

        # ----------------- Tabs Layout -----------------
        tab1, tab2, tab3, tab4 = st.tabs(["👀 Preview", "📑 Summary", "📈 Visualizations", "🔗 Correlation"])

        # Tab 1: Preview
        with tab1:
            st.subheader("Filtered data preview")
            st.dataframe(df_filtered.head(10), use_container_width=True)
            csv_cleaned = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download cleaned CSV", data=csv_cleaned, file_name="cleaned_data.csv", mime="text/csv")

        # Tab 2: Summary
        with tab2:
            st.subheader("Column summary & missing values")
            col_summary = df_filtered.dtypes.to_frame('Data Type')
            col_summary['Non-Null Count'] = df_filtered.count()
            col_summary['Missing Count'] = df_filtered.isnull().sum()
            col_summary['Missing %'] = (col_summary['Missing Count'] / len(df_filtered) * 100).round(2)
            col_summary['Unique Values'] = df_filtered.nunique()
            st.dataframe(col_summary, use_container_width=True)

            if len(numeric_cols) > 0:
                stats = df_filtered[numeric_cols].describe().T
                csv_stats = stats.to_csv().encode('utf-8')
                st.download_button("📥 Download summary statistics", data=csv_stats, file_name="summary_statistics.csv", mime="text/csv")

            if df_filtered.isnull().sum().sum() > 0:
                st.subheader("Missing values heatmap")
                fig, ax = plt.subplots(figsize=(10,6))
                sns.heatmap(df_filtered.isnull(), cbar=False, cmap="magma", ax=ax)
                st.pyplot(fig)

        # Tab 3: Visualizations
        with tab3:
            st.subheader("Numeric & categorical visualizations")
            c1, c2 = st.columns(2)

            with c1:
                if len(numeric_cols) > 0:
                    selected_num_col = st.selectbox("Select numeric column", numeric_cols, key="numviz")
                    fig = px.histogram(df_filtered, x=selected_num_col, nbins=30, color_discrete_sequence=["#FF4B4B"])
                    st.plotly_chart(fig, use_container_width=True)

            with c2:
                if len(categorical_cols) > 0:
                    selected_cat_col = st.selectbox("Select categorical column", categorical_cols, key="catviz")
                    df_counts = df_filtered[selected_cat_col].astype(str).value_counts().reset_index()
                    df_counts.columns = [selected_cat_col, 'count']
                    fig = px.bar(
                        df_counts,
                        x=selected_cat_col,
                        y='count',
                        color=selected_cat_col,
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # Tab 4: Correlation
        with tab4:
            if len(numeric_cols) > 1:
                st.subheader("Correlation heatmap")
                corr = df_filtered[numeric_cols].corr(numeric_only=True)
                fig, ax = plt.subplots(figsize=(10,8))
                sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax)
                st.pyplot(fig)

                st.subheader("Interactive correlation matrix")
                fig2 = px.imshow(corr, text_auto=True, color_continuous_scale="Viridis", template="plotly_dark")
                st.plotly_chart(fig2, use_container_width=True)

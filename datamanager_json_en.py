# Tool for Viewing, Analyzing, and Filtering JSON Files
# Requirements: pip install streamlit pandas orjson matplotlib seaborn plotly textstat nltk langdetect textblob
# Save the script as datamanager_json.py
# Run: streamlit run datamanager_json.py
#
# @2024 Jan Schachtschabel
# Licence: Apache 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

import streamlit as st
import pandas as pd
import orjson
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import textstat
import re
import nltk
from nltk import ngrams
from collections import Counter, defaultdict
from langdetect import detect, LangDetectException
from textblob import TextBlob
import time
from itertools import cycle
from functools import lru_cache
from io import BytesIO

# Download NLTK data
nltk.download('punkt')

# Function to recursively flatten JSON
def flatten_json(y):
    out = {}
    def flatten(x, name=''):
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], f'{name}{a}_')
        elif isinstance(x, list):
            # Join list items with a comma
            out[name[:-1]] = ', '.join(map(str, x))
        else:
            out[name[:-1]] = x
    flatten(y)
    return out

@st.cache_data(show_spinner=False)
def process_uploaded_file(uploaded_file):
    try:
        content = uploaded_file.read()
        data = orjson.loads(content)
        
        # Store the original JSON data
        st.session_state.original_json = data
        st.session_state.selected_file = uploaded_file.name

        # Flatten the JSON data for display and filtering
        records = []
        if isinstance(data, list):
            for record in data:
                flat_record = flatten_json(record)
                records.append(flat_record)
        elif isinstance(data, dict):
            flat_record = flatten_json(data)
            records.append(flat_record)
        else:
            st.error("Unsupported JSON structure.")
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        st.success(f"JSON data successfully converted to DataFrame. Columns: {len(df.columns)}")
        st.write(f"Number of records: {len(df)}")
        return df
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return pd.DataFrame()

def merge_similar_fields(df):
    pattern = re.compile(r'^(.*?)(?:_\d+)?$')
    base_columns = {}
    for col in df.columns:
        match = pattern.match(col)
        if match:
            base_name = match.group(1)
            if base_name not in base_columns:
                base_columns[base_name] = []
            base_columns[base_name].append(col)
    
    for base, cols in base_columns.items():
        if len(cols) > 1:
            df[base] = df[cols].apply(lambda row: ', '.join(row.dropna().astype(str)), axis=1)
            df.drop(columns=cols, inplace=True)
    
    return df

def calculate_fill_status(df):
    fill_status = df.notnull().mean() * 100
    fill_status = fill_status.sort_values(ascending=False)
    return fill_status

def get_all_fields(data, parent_key='', fields=None):
    if fields is None:
        fields = set()
    if isinstance(data, dict):
        for key, value in data.items():
            full_key = f'{parent_key}.{key}' if parent_key else key
            fields.add(full_key)
            if isinstance(value, dict):
                get_all_fields(value, full_key, fields)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        get_all_fields(item, full_key, fields)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                get_all_fields(item, parent_key, fields)
    return fields

def load_json(file_path):
    with open(file_path, "rb") as f:
        return orjson.loads(f.read())

def save_json(data, file_path):
    with open(file_path, "wb") as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))

def preview_data(data, index=0):
    if 0 <= index < len(data):
        return data[index]
    return {}

def get_nested_value(data, path):
    keys = path.split(".")
    for key in keys:
        if isinstance(data, list):
            # Extract values for the key from each dict in the list
            next_data = []
            for item in data:
                if isinstance(item, dict) and key in item:
                    next_data.append(item[key])
            data = next_data if next_data else None
        elif isinstance(data, dict):
            data = data.get(key)
        else:
            return None
        if data is None:
            return None
    # Flatten the list if it's a list of lists
    if isinstance(data, list):
        flat_data = []
        for item in data:
            if isinstance(item, list):
                flat_data.extend(item)
            else:
                flat_data.append(item)
        return flat_data
    return data

def is_field_empty(value):
    """Checks if a field is considered 'empty' (e.g., None, empty strings, lists, dicts)."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, list) and len(value) == 0:
        return True
    if isinstance(value, dict) and len(value) == 0:
        return True
    return False

def remove_fields(data, fields_to_remove):
    for item in data:
        for field in fields_to_remove:
            keys = field.split(".")
            current_dict = item
            for key in keys[:-1]:
                if key in current_dict and isinstance(current_dict[key], dict):
                    current_dict = current_dict[key]
                else:
                    current_dict = {}
                    break
            if keys[-1] in current_dict:
                del current_dict[keys[-1]]
    return data

def current_timestamp():
    return time.strftime("%Y%m%d_%H%M%S")

def text_analysis(df, text_field, min_chars=0):
    try:
        texts = df[text_field].dropna().astype(str)
    except KeyError:
        st.error(f"Field '{text_field}' does not exist.")
        return

    if min_chars > 0:
        texts = texts[texts.str.len() >= min_chars]

    if 'text_index' not in st.session_state:
        st.session_state.text_index = 1

    st.subheader("Browse Texts")
    if not texts.empty:
        # Use Streamlit's native functions for navigation
        st.markdown("### Text Preview")
        max_index = len(texts)
        st.session_state.text_index = st.number_input(
            "Record Number",
            min_value=1,
            max_value=max_index,
            value=st.session_state.text_index if st.session_state.text_index <= max_index else max_index,
            step=1,
            key='text_navigation'
        )
        current_text = texts.iloc[st.session_state.text_index - 1]
        st.text_area("Text", value=current_text, height=200, key='text_display')

        st.write(f"**Record Number:** {st.session_state.text_index}")

        st.subheader("Text Statistics")
        num_chars = texts.str.len().sum()
        num_words = texts.apply(lambda x: len(x.split())).sum()
        avg_chars = texts.str.len().mean()
        avg_words = texts.apply(lambda x: len(x.split())).mean()

        st.write(f"**Total Characters:** {num_chars}")
        st.write(f"**Total Words:** {num_words}")
        st.write(f"**Average Characters per Text:** {avg_chars:.2f}")
        st.write(f"**Average Words per Text:** {avg_words:.2f}")

        # **Readability Metrics**
        st.subheader("Readability Metrics")
        st.markdown("""
        **Flesch Reading Ease:** Measures how easy a text is to read. Higher scores indicate easier readability.

        **Flesch-Kincaid Grade:** Indicates the U.S. school grade level required to understand the text.

        **Gunning Fog Index:** Estimates the years of formal education needed to understand the text on first reading.

        **SMOG Index:** Estimates the years of education needed to understand a piece of writing based on the number of polysyllabic words.
        """)
        readability_df = pd.DataFrame({
            'Flesch Reading Ease': texts.apply(textstat.flesch_reading_ease),
            'Flesch-Kincaid Grade': texts.apply(textstat.flesch_kincaid_grade),
            'Gunning Fog Index': texts.apply(textstat.gunning_fog),
            'SMOG Index': texts.apply(textstat.smog_index)
        })
        readability_summary = readability_df.mean().round(2)
        st.write(readability_summary.to_frame(name='Average').T)

        st.markdown("""
        **Interpretation of Readability Metrics:**
        - **Flesch Reading Ease:** Scores between 60-70 are considered easily understandable by most adults.
        - **Flesch-Kincaid Grade:** A score of 8 means that a student in 8th grade should be able to understand the text.
        - **Gunning Fog Index:** A score of 12 corresponds to the reading level of a high school graduate.
        - **SMOG Index:** Estimates the years of education required to understand the text.
        """)

        # **Sentiment Analysis**
        st.subheader("Sentiment Analysis")
        st.markdown("""
        **Sentiment Analysis:** Determines the emotional tone of the texts. Categories include:
        - **Positive:** Text expresses positive emotions.
        - **Negative:** Text expresses negative emotions.
        - **Neutral:** Text expresses neither positive nor negative emotions.
        """)
        sentiments = texts.apply(lambda x: TextBlob(x).sentiment.polarity)
        sentiment_counts = sentiments.apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral')).value_counts()

        sentiment_counts_df = sentiment_counts.reset_index()
        sentiment_counts_df.columns = ['Sentiment', 'Number of Texts']

        fig2 = px.bar(
            sentiment_counts_df,
            x='Sentiment',
            y='Number of Texts',
            labels={'Number of Texts': 'Number of Texts'},
            title="Sentiment Category Distribution",
            hover_data={'Number of Texts': True}
        )
        fig2.update_traces(marker_color='blue')
        fig2.update_layout(xaxis_title='Sentiment', yaxis_title='Number of Texts')
        st.plotly_chart(fig2, use_container_width=True, key='sentiment_plot')

        st.markdown("""
        **Interpretation of Sentiment Analysis:**
        - **Positive:** A high percentage of positive texts may indicate an optimistic sentiment in the data.
        - **Negative:** A high percentage of negative texts may indicate challenges or criticisms in the data.
        - **Neutral:** A high percentage of neutral texts indicates factual or informative content.
        """)

        # **Language Detection**
        st.subheader("Language Detection")
        st.markdown("""
        **Language Detection:** Identifies the language of the texts to ensure that all texts are written in the expected language.
        """)
        def detect_language(text):
            try:
                return detect(text)
            except LangDetectException:
                return "Unknown"

        languages = texts.apply(detect_language)
        language_counts = languages.value_counts()

        language_counts_df = language_counts.reset_index()
        language_counts_df.columns = ['Language', 'Number of Texts']

        fig3 = px.bar(
            language_counts_df,
            x='Language',
            y='Number of Texts',
            labels={'Number of Texts': 'Number of Texts'},
            title="Detected Language Distribution",
            hover_data={'Number of Texts': True}
        )
        fig3.update_traces(marker_color='orange')
        fig3.update_layout(xaxis_title='Language', yaxis_title='Number of Texts')
        st.plotly_chart(fig3, use_container_width=True, key='language_plot')

        st.markdown("""
        **Interpretation of Language Detection:**
        - **Language:** The detected languages provide insight into which languages are predominant in the text data.
        - **Unknown:** A high percentage of "Unknown" may indicate unclear or mixed language content.
        """)

        # **Lexical Diversity**
        st.subheader("Lexical Diversity")
        st.markdown("""
        **Lexical Diversity (Type-Token Ratio, TTR):** Measures the variety of vocabulary used. A higher TTR indicates greater word diversity, suggesting richer and more varied language in the text.
        """)
        def type_token_ratio(text):
            tokens = text.split()
            types = set(tokens)
            return len(types) / len(tokens) if len(tokens) > 0 else 0

        df['TTR'] = texts.apply(type_token_ratio)
        ttr_summary = df['TTR'].describe()
        st.write("**Statistics of Type-Token Ratio (TTR):**")
        st.write(ttr_summary)

        fig5 = px.histogram(
            df,
            x='TTR',
            nbins=20,
            title="Type-Token Ratio (TTR) Distribution",
            labels={'TTR': 'TTR', 'count': 'Number of Texts'},
            opacity=0.75
        )
        fig5.update_traces(marker_color='green')
        st.plotly_chart(fig5, use_container_width=True, key='ttr_plot')

        st.markdown("""
        **Interpretation of Lexical Diversity:**
        - **Higher TTR:** Greater variety in vocabulary, indicating more diverse and rich texts.
        - **Lower TTR:** Less variety in vocabulary, indicating repetitive or monotonous language.
        """)

        # **Duplicate Detection**
        st.subheader("Duplicate Detection")
        st.markdown("""
        **Duplicate Detection:** Identifies duplicate or nearly duplicate texts to avoid redundancies in the data.
        """)
        duplicate_counts = df[text_field].duplicated().sum()
        st.write(f"**Number of Duplicate Texts:** {duplicate_counts}")

        if duplicate_counts > 0:
            duplicates = df[df[text_field].duplicated(keep=False)]
            st.write("**Duplicate Texts:**")
            st.write(duplicates[[text_field]])

        # **N-Gram Analysis**
        st.subheader("N-Gram Analysis")
        st.markdown("""
        **N-Gram Analysis:** Analyzes frequently occurring phrases (bigrams) to identify common expressions or themes.
        """)
        def get_ngrams(text, n=2):
            tokens = nltk.word_tokenize(text)
            return list(ngrams(tokens, n))

        bigrams = texts.apply(lambda x: get_ngrams(x, 2)).explode()
        bigram_counts = Counter(bigrams).most_common(20)
        bigram_df = pd.DataFrame(bigram_counts, columns=['Bigram', 'Count'])
        bigram_df['Bigram'] = bigram_df['Bigram'].apply(lambda x: ' '.join(x))

        fig6 = px.bar(
            bigram_df,
            x='Count',
            y='Bigram',
            orientation='h',
            labels={'Count': 'Number of Occurrences'},
            title="Top 20 Bigrams",
            hover_data={'Count': True}
        )
        fig6.update_traces(marker_color='magenta')
        st.plotly_chart(fig6, use_container_width=True, key='bigram_plot')

        st.markdown("""
        **Interpretation of N-Gram Analysis:**
        - **Frequent Bigrams:** The most frequently occurring bigrams can indicate common phrases or themes in the texts.
        """)

@st.cache_data
def get_unique_values(data, field):
    unique_vals = set()
    for item in data:
        value = get_nested_value(item, field)
        if isinstance(value, list):
            unique_vals.update([str(v).strip() for v in value if isinstance(v, (int, float, str))])
        elif isinstance(value, str):
            unique_vals.update([v.strip() for v in value.split(', ') if v.strip()])
        elif value is not None:
            unique_vals.add(str(value).strip())
    return sorted(unique_vals)

def json_filter_tab(df):
    st.header("🔄 Data Filter")

    # Access the original JSON data
    if 'original_json' not in st.session_state:
        st.error("Original JSON data is not available.")
        return
    
    data = st.session_state.original_json
    all_fields = set(get_all_fields(data))

    # Filter 1: Empty Fields (default inactive)
    empty_field_filter_active = st.checkbox("🚫 Filter Empty Fields", value=False)
    if empty_field_filter_active:
        selected_empty_fields = st.multiselect(
            "Select fields to check for empty values",
            options=list(all_fields),
            default=[]
        )
        st.warning("This filter removes records where selected fields have no values.")

    # Filter 2: Field-Value Combinations with Operators and Autocomplete
    field_value_filter_active = st.checkbox("🔍 Filter Field-Value Combinations")
    if field_value_filter_active:
        st.warning("This filter removes records that do not match the specified field-value combinations.")
        field_value_filters = []
        field_value_count = st.number_input("Number of Field-Value Combinations", min_value=1, value=1, step=1)
        operators = ["=", "!=", ">", "<", ">=", "<="]
        operator_map = {
            "=": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
            ">": lambda a, b: a > b,
            "<": lambda a, b: a < b,
            ">=": lambda a, b: a >= b,
            "<=": lambda a, b: a <= b
        }
        
        for i in range(int(field_value_count)):
            col1, col2, col3 = st.columns(3)
            with col1:
                field = st.selectbox(f"Field {i+1}", options=list(all_fields), key=f"filter_field_{i}")
            with col2:
                operator = st.selectbox(f"Operator {i+1}", options=operators, key=f"filter_operator_{i}")
            with col3:
                value = st.text_input(f"Value {i+1}", key=f"filter_value_{i}")
            field_value_filters.append((field, operator, value))

    # Filter 3: Minimum Character Length
    length_filter_active = st.checkbox("✂️ Filter by Minimum Character Length")
    if length_filter_active:
        selected_length_fields = st.multiselect("Select fields to filter by character length", options=list(all_fields))
        min_length = st.number_input("Minimum Character Length", min_value=1, value=30)
        st.warning("This filter removes records where the length of the selected fields is shorter than the specified minimum length.")

    # Filter 4: Balancing
    balancing_filter_active = st.checkbox("⚖️ Balancing Filter")
    if balancing_filter_active:
        selected_balancing_fields = st.multiselect("Select fields for balancing", options=list(all_fields))
        total_count = st.number_input("Total number of records after balancing", min_value=1, value=100)
        st.warning("This filter reduces the data to a total number of records and distributes them as evenly as possible across the selected fields.")

    # Filter 5: Remove Fields from JSON
    remove_fields_filter_active = st.checkbox("🗑️ Remove Fields from JSON")
    if remove_fields_filter_active:
        fields_to_remove = st.multiselect("Select fields to remove", options=list(all_fields), default=[])
        st.warning("This filter removes the selected fields from the records.")

    # Filter 6: Remove Duplicates
    duplicate_filter_active = st.checkbox("🔁 Remove Duplicates")
    if duplicate_filter_active:
        duplicate_fields = st.multiselect("Select fields to base duplicate removal on", options=list(all_fields), default=[])
        st.warning("This filter removes duplicate records based on the selected fields.")

    # Status messages and debugging information
    if st.button("✅ Apply Filters and Save"):
        st.info("Starting filter process...")
        filtered_data = data.copy()  # Copy original JSON data

        # Filter Empty Fields
        if empty_field_filter_active and selected_empty_fields:
            st.info("🚫 Filtering empty fields...")
            filtered_data = [
                item for item in filtered_data 
                if all(
                    (field_value := get_nested_value(item, field)) is not None 
                    and not is_field_empty(field_value)
                    for field in selected_empty_fields
                )
            ]
            st.write(f"Number of records after filtering empty fields: {len(filtered_data)}")

        # Filter Field-Value Combinations
        if field_value_filter_active and field_value_filters:
            st.info("🔍 Filtering field-value combinations...")
            for field, operator, value in field_value_filters:
                op_func = operator_map[operator]
                try:
                    # Attempt to convert the value to a numeric type if possible
                    try:
                        value_converted = float(value)
                    except ValueError:
                        value_converted = value
                    
                    filtered_data = [
                        item for item in filtered_data
                        if (field_value := get_nested_value(item, field)) is not None and (
                            (isinstance(field_value, list) and any(
                                isinstance(v, (int, float, str)) and op_func(v, value_converted) for v in field_value
                            )) or (isinstance(field_value, (int, float, str)) and op_func(field_value, value_converted))
                        )
                    ]
                except TypeError:
                    st.error(f"The value in field '{field}' cannot be compared using operator '{operator}'.")
            st.write(f"Number of records after field-value filtering: {len(filtered_data)}")

        # Filter by Character Length
        if length_filter_active and selected_length_fields:
            st.info("✂️ Filtering by character length...")
            filtered_data = [
                item for item in filtered_data 
                if all(
                    (field_value := get_nested_value(item, field)) is not None 
                    and (
                        (isinstance(field_value, str) and len(field_value) >= min_length)
                        or (isinstance(field_value, list) and any(isinstance(v, str) and len(v) >= min_length for v in field_value))
                    )
                    for field in selected_length_fields
                )
            ]
            st.write(f"Number of records after filtering by minimum character length: {len(filtered_data)}")

        # Balancing Filter
        if balancing_filter_active and selected_balancing_fields:
            st.info("⚖️ Starting balancing...")
            field_groups = defaultdict(list)
            for item in filtered_data:
                # Create a hashable key by converting lists to tuples
                key = tuple(
                    tuple(get_nested_value(item, field)) if isinstance(get_nested_value(item, field), list) else get_nested_value(item, field)
                    for field in selected_balancing_fields
                )
                field_groups[key].append(item)
            
            balanced_data = []
            groups = list(field_groups.values())
            if groups:
                group_cycle = cycle(groups)
                while len(balanced_data) < total_count and groups:
                    try:
                        group = next(group_cycle)
                        if group:
                            balanced_data.append(group.pop(0))
                            if not group:
                                groups.remove(group)
                                group_cycle = cycle(groups)
                    except StopIteration:
                        break
            filtered_data = balanced_data[:total_count]
            st.write(f"Number of records after balancing: {len(filtered_data)}")

        # Remove Duplicates
        if duplicate_filter_active and duplicate_fields:
            st.info("🔁 Removing duplicates...")
            initial_count = len(filtered_data)
            if duplicate_fields:
                filtered_df = pd.DataFrame(filtered_data)
                filtered_df = filtered_df.drop_duplicates(subset=duplicate_fields, keep='first')
                filtered_data = filtered_df.to_dict(orient='records')
                duplicate_removed = initial_count - len(filtered_data)
                st.write(f"Duplicates removed: {duplicate_removed}")
                st.write(f"Number of remaining records: {len(filtered_data)}")
            else:
                st.warning("Please select at least one field for duplicate filtering.")

        # Remove Fields
        if remove_fields_filter_active and fields_to_remove:
            st.info("🗑️ Removing fields...")
            filtered_data = remove_fields(filtered_data, fields_to_remove)
            st.write(f"Number of records after removing fields: {len(filtered_data)} (Count remains the same)")

        # Save filtered data with name suffixes and offer download
        timestamp = current_timestamp()
        filters_applied = []
        if empty_field_filter_active and selected_empty_fields:
            filters_applied.append("emptyfields")
        if field_value_filter_active and field_value_filters:
            filters_applied.append("fieldvalue")
        if length_filter_active and selected_length_fields:
            filters_applied.append(f"minlength{min_length}")
        if balancing_filter_active and selected_balancing_fields:
            filters_applied.append("balancing")
        if duplicate_filter_active and duplicate_fields:
            filters_applied.append("duplicates")
        if remove_fields_filter_active and fields_to_remove:
            filters_applied.append("removefields")
        
        filters_suffix = '_'.join(filters_applied) if filters_applied else "filtered"
        selected_file = st.session_state.get('selected_file', 'output.json')
        base_name = os.path.splitext(selected_file)[0]
        output_filename = f"{base_name}_{filters_suffix}_{timestamp}.json"
        
        # Convert the filtered data to JSON bytes
        filtered_json_bytes = orjson.dumps(filtered_data, option=orjson.OPT_INDENT_2)
        
        # Provide a download button
        st.download_button(
            label="📥 Download Filtered JSON File",
            data=filtered_json_bytes,
            file_name=output_filename,
            mime="application/json"
        )
        st.write(f"Number of records after filtering: {len(filtered_data)}")
        st.success(f"Filtered data is available for download as: {output_filename}")

def data_viewer_tab(df):
    st.header("📁 Data Viewer")

    with st.expander("🔍 Record Preview"):
        if 'viewer_index' not in st.session_state:
            st.session_state.viewer_index = 0
        current_record = preview_data(df.to_dict(orient='records'), st.session_state.viewer_index)
        st.json(current_record)
        
        # Display current record number after button click
        st.write(f"**Record Number:** {st.session_state.viewer_index + 1}")

        # Buttons to navigate (now below the elements)
        col_prev, col_next = st.columns([1,1])
        with col_prev:
            if st.button("⬅️ Previous Record", key='prev_viewer'):
                if st.session_state.viewer_index > 0:
                    st.session_state.viewer_index -= 1
        with col_next:
            if st.button("➡️ Next Record", key='next_viewer'):
                if st.session_state.viewer_index < len(df) - 1:
                    st.session_state.viewer_index += 1

    st.subheader("📊 Display Records")
    st.markdown("**Note:** Large datasets may take a long time to display.")
    col_start, col_end = st.columns(2)
    with col_start:
        start_num = st.number_input("Start Record Number", min_value=1, value=1, step=1, key='start_num')
    with col_end:
        end_num = st.number_input("End Record Number", min_value=1, value=min(len(df), 10), step=1, key='end_num')
    st.write("")  # Blank line
    if st.button("🔄 Show Data"):
        if end_num < start_num:
            st.error("End number must be greater than or equal to start number.")
        elif end_num > len(df):
            st.error(f"End number cannot be greater than the number of records ({len(df)}).")
        else:
            st.write(df.iloc[start_num-1:end_num])

def werteverteilung_tab(df):
    st.header("📈 Value Distribution")
    metadata_fields = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if not metadata_fields:
        st.write("No metadata fields found.")
    else:
        search_field = st.text_input("Search Field", "", key='metadata_search')
        if search_field:
            filtered_fields = [field for field in metadata_fields if search_field.lower() in field.lower()]
        else:
            filtered_fields = metadata_fields
        if filtered_fields:
            selected_fields = st.multiselect("Select Metadata Fields for Visualization", filtered_fields, key='metadata_select_multi')
            if selected_fields:
                for field in selected_fields:
                    # Handle multiple values by splitting
                    value_series = df[field].dropna().astype(str).str.split(', ').explode()
                    value_counts = value_series.value_counts().head(20)
                    st.write(value_counts.to_frame().reset_index().rename(columns={'index': field, field: 'Count'}))
                    
                    # Truncate labels for readability
                    df_counts = value_counts.to_frame().reset_index()
                    df_counts.columns = [field, 'Count']
                    df_counts[field] = df_counts[field].apply(lambda x: x[:20] + '...' if len(x) > 20 else x)
                    
                    fig = px.bar(
                        df_counts,
                        x='Count',
                        y=field,
                        orientation='h',
                        labels={'Count': 'Number of Occurrences', field: 'Field'},
                        title="",  # Remove the title above the chart
                        hover_data={'Count': True}
                    )
                    fig.update_traces(marker_color='blue')
                    st.plotly_chart(fig, use_container_width=True, key=f'value_distribution_{field}')
        else:
            st.write("No fields found matching the search term.")

def fuellstandsanalyse_tab(df):
    st.header("📊 Fill Level Analysis")
    st.write("Filter data based on fields and values and analyze the fill level of metadata.")

    # Selection of fields and values for filtering
    st.subheader("🔍 Select Filters")
    selected_fill_fields = st.multiselect(
        "Select fields for filtering",
        options=df.columns.tolist(),
        default=[]
    )
    fill_field_values = {}
    for field in selected_fill_fields:
        unique_values = get_unique_values(st.session_state.original_json, field)
        selected_values = st.multiselect(f"Select values for {field}", unique_values, default=[], key=f"fill_{field}")
        fill_field_values[field] = selected_values

    # Option to apply filters jointly or separately
    join_option = st.radio(
        "How should the filters be applied?",
        options=["View Separately", "Create Combined Data Set"],
        index=0
    )

    # Selection of fields to display fill levels
    st.subheader("📈 Display Fill Levels for:")
    display_fill_fields = st.multiselect(
        "Select metadata fields to display fill levels",
        options=df.columns.tolist(),
        default=[]
    )

    if st.button("🔄 Analyze Fill Levels"):
        st.info("Starting fill level analysis...")
        if selected_fill_fields and any(fill_field_values[field] for field in selected_fill_fields):
            if join_option == "Create Combined Data Set":
                # Filter data that meets all selected field-value combinations
                filtered_df = df.copy()
                for field, values in fill_field_values.items():
                    if values:
                        filtered_df = filtered_df[filtered_df[field].isin(values)]
                subsets = {"Combined Data Set": filtered_df}
            else:
                # Each field-value combination as a separate subset
                subsets = {}
                for field, values in fill_field_values.items():
                    if values:
                        for value in values:
                            subset_name = f"{field} = {value}"
                            subsets[subset_name] = df[df[field].isin([value])]
        else:
            # No filters applied, a single subset
            subsets = {"All Data": df}

        if display_fill_fields:
            # Limit the number of charts to maintain clarity
            max_columns = 2  # Changed from 3 to 2 columns
            num_subsets = len(subsets)
            num_cols = min(max_columns, num_subsets)
            cols = st.columns(num_cols)
            
            for idx, (subset_name, subset_df) in enumerate(subsets.items()):
                col = cols[idx % num_cols]
                with col:
                    # Add a note above the chart
                    st.markdown(f"**Filter:** {subset_name}")
                    
                    # Remove the title above the charts
                    fill_status = subset_df[display_fill_fields].notnull().mean() * 100
                    fill_status = fill_status.sort_values(ascending=False)

                    # Dynamic adjustment of chart size
                    num_bars = len(fill_status)
                    if num_bars == 1:
                        fig_height = 400
                    else:
                        fig_height = max(400, num_bars * 50)
                    
                    # Prepare data for dual-colored bars
                    fill_percentage = fill_status
                    empty_percentage = 100 - fill_status
                    fill_data = pd.DataFrame({
                        'Metadata Field': fill_status.index,
                        'Filled (%)': fill_percentage.values,
                        'Empty (%)': empty_percentage.values
                    })

                    # Melt the DataFrame for stacked bar charts
                    fill_data_melted = fill_data.melt(id_vars='Metadata Field', value_vars=['Filled (%)', 'Empty (%)'], var_name='Status', value_name='Percentage')

                    fig = px.bar(
                        fill_data_melted,
                        x='Percentage',
                        y='Metadata Field',
                        color='Status',
                        orientation='h',
                        title="",  # Remove the title above the chart
                        labels={'Percentage': 'Percentage (%)', 'Metadata Field': 'Metadata Field'},
                        hover_data={'Percentage': True, 'Status': True}
                    )
                    fig.update_layout(barmode='stack', height=fig_height, showlegend=True)
                    st.plotly_chart(fig, use_container_width=True, key=f'fill_level_plot_{idx}')
        else:
            st.warning("Please select at least one field to display fill levels.")

def text_analysis_tab(df):
    st.header("📝 Text Analysis")
    text_fields = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if not text_fields:
        st.write("No text fields found.")
    else:
        search_text_field = st.text_input("Search Field", "", key='text_search')
        if search_text_field:
            filtered_text_fields = [field for field in text_fields if search_text_field.lower() in field.lower()]
        else:
            filtered_text_fields = text_fields
        if filtered_text_fields:
            selected_text_field = st.selectbox("Select a Metadata Field with Text", filtered_text_fields, key='text_select')
            if selected_text_field:
                min_chars = st.number_input("Minimum Number of Characters to Filter", min_value=0, value=0, step=1, key='text_min_chars')
                text_analysis(df, selected_text_field, min_chars)
        else:
            st.write("No text fields found matching the search term.")

def json_filter_tab(df):
    st.header("🔄 Data Filter")

    # Access the original JSON data
    if 'original_json' not in st.session_state:
        st.error("Original JSON data is not available.")
        return
    
    data = st.session_state.original_json
    all_fields = set(get_all_fields(data))

    # Filter 1: Empty Fields (default inactive)
    empty_field_filter_active = st.checkbox("🚫 Filter Empty Fields", value=False)
    if empty_field_filter_active:
        selected_empty_fields = st.multiselect(
            "Select fields to check for empty values",
            options=list(all_fields),
            default=[]
        )
        st.warning("This filter removes records where selected fields have no values.")

    # Filter 2: Field-Value Combinations with Operators and Autocomplete
    field_value_filter_active = st.checkbox("🔍 Filter Field-Value Combinations")
    if field_value_filter_active:
        st.warning("This filter removes records that do not match the specified field-value combinations.")
        field_value_filters = []
        field_value_count = st.number_input("Number of Field-Value Combinations", min_value=1, value=1, step=1)
        operators = ["=", "!=", ">", "<", ">=", "<="]
        operator_map = {
            "=": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
            ">": lambda a, b: a > b,
            "<": lambda a, b: a < b,
            ">=": lambda a, b: a >= b,
            "<=": lambda a, b: a <= b
        }
        
        for i in range(int(field_value_count)):
            col1, col2, col3 = st.columns(3)
            with col1:
                field = st.selectbox(f"Field {i+1}", options=list(all_fields), key=f"filter_field_{i}")
            with col2:
                operator = st.selectbox(f"Operator {i+1}", options=operators, key=f"filter_operator_{i}")
            with col3:
                value = st.text_input(f"Value {i+1}", key=f"filter_value_{i}")
            field_value_filters.append((field, operator, value))

    # Filter 3: Minimum Character Length
    length_filter_active = st.checkbox("✂️ Filter by Minimum Character Length")
    if length_filter_active:
        selected_length_fields = st.multiselect("Select fields to filter by character length", options=list(all_fields))
        min_length = st.number_input("Minimum Character Length", min_value=1, value=30)
        st.warning("This filter removes records where the length of the selected fields is shorter than the specified minimum length.")

    # Filter 4: Balancing
    balancing_filter_active = st.checkbox("⚖️ Balancing Filter")
    if balancing_filter_active:
        selected_balancing_fields = st.multiselect("Select fields for balancing", options=list(all_fields))
        total_count = st.number_input("Total number of records after balancing", min_value=1, value=100)
        st.warning("This filter reduces the data to a total number of records and distributes them as evenly as possible across the selected fields.")

    # Filter 5: Remove Fields from JSON
    remove_fields_filter_active = st.checkbox("🗑️ Remove Fields from JSON")
    if remove_fields_filter_active:
        fields_to_remove = st.multiselect("Select fields to remove", options=list(all_fields), default=[])
        st.warning("This filter removes the selected fields from the records.")

    # Filter 6: Remove Duplicates
    duplicate_filter_active = st.checkbox("🔁 Remove Duplicates")
    if duplicate_filter_active:
        duplicate_fields = st.multiselect("Select fields to base duplicate removal on", options=list(all_fields), default=[])
        st.warning("This filter removes duplicate records based on the selected fields.")

    # Status messages and debugging information
    if st.button("✅ Apply Filters and Save"):
        st.info("Starting filter process...")
        filtered_data = data.copy()  # Copy original JSON data

        # Filter Empty Fields
        if empty_field_filter_active and selected_empty_fields:
            st.info("🚫 Filtering empty fields...")
            filtered_data = [
                item for item in filtered_data 
                if all(
                    (field_value := get_nested_value(item, field)) is not None 
                    and not is_field_empty(field_value)
                    for field in selected_empty_fields
                )
            ]
            st.write(f"Number of records after filtering empty fields: {len(filtered_data)}")

        # Filter Field-Value Combinations
        if field_value_filter_active and field_value_filters:
            st.info("🔍 Filtering field-value combinations...")
            for field, operator, value in field_value_filters:
                op_func = operator_map[operator]
                try:
                    # Attempt to convert the value to a numeric type if possible
                    try:
                        value_converted = float(value)
                    except ValueError:
                        value_converted = value
                    
                    filtered_data = [
                        item for item in filtered_data
                        if (field_value := get_nested_value(item, field)) is not None and (
                            (isinstance(field_value, list) and any(
                                isinstance(v, (int, float, str)) and op_func(v, value_converted) for v in field_value
                            )) or (isinstance(field_value, (int, float, str)) and op_func(field_value, value_converted))
                        )
                    ]
                except TypeError:
                    st.error(f"The value in field '{field}' cannot be compared using operator '{operator}'.")
            st.write(f"Number of records after field-value filtering: {len(filtered_data)}")

        # Filter by Character Length
        if length_filter_active and selected_length_fields:
            st.info("✂️ Filtering by character length...")
            filtered_data = [
                item for item in filtered_data 
                if all(
                    (field_value := get_nested_value(item, field)) is not None 
                    and (
                        (isinstance(field_value, str) and len(field_value) >= min_length)
                        or (isinstance(field_value, list) and any(isinstance(v, str) and len(v) >= min_length for v in field_value))
                    )
                    for field in selected_length_fields
                )
            ]
            st.write(f"Number of records after filtering by minimum character length: {len(filtered_data)}")

        # Balancing Filter
        if balancing_filter_active and selected_balancing_fields:
            st.info("⚖️ Starting balancing...")
            field_groups = defaultdict(list)
            for item in filtered_data:
                # Create a hashable key by converting lists to tuples
                key = tuple(
                    tuple(get_nested_value(item, field)) if isinstance(get_nested_value(item, field), list) else get_nested_value(item, field)
                    for field in selected_balancing_fields
                )
                field_groups[key].append(item)
            
            balanced_data = []
            groups = list(field_groups.values())
            if groups:
                group_cycle = cycle(groups)
                while len(balanced_data) < total_count and groups:
                    try:
                        group = next(group_cycle)
                        if group:
                            balanced_data.append(group.pop(0))
                            if not group:
                                groups.remove(group)
                                group_cycle = cycle(groups)
                    except StopIteration:
                        break
            filtered_data = balanced_data[:total_count]
            st.write(f"Number of records after balancing: {len(filtered_data)}")

        # Remove Duplicates
        if duplicate_filter_active and duplicate_fields:
            st.info("🔁 Removing duplicates...")
            initial_count = len(filtered_data)
            if duplicate_fields:
                filtered_df = pd.DataFrame(filtered_data)
                filtered_df = filtered_df.drop_duplicates(subset=duplicate_fields, keep='first')
                filtered_data = filtered_df.to_dict(orient='records')
                duplicate_removed = initial_count - len(filtered_data)
                st.write(f"Duplicates removed: {duplicate_removed}")
                st.write(f"Number of remaining records: {len(filtered_data)}")
            else:
                st.warning("Please select at least one field for duplicate filtering.")

        # Remove Fields
        if remove_fields_filter_active and fields_to_remove:
            st.info("🗑️ Removing fields...")
            filtered_data = remove_fields(filtered_data, fields_to_remove)
            st.write(f"Number of records after removing fields: {len(filtered_data)} (Count remains the same)")

        # Save filtered data with name suffixes and offer download
        timestamp = current_timestamp()
        filters_applied = []
        if empty_field_filter_active and selected_empty_fields:
            filters_applied.append("emptyfields")
        if field_value_filter_active and field_value_filters:
            filters_applied.append("fieldvalue")
        if length_filter_active and selected_length_fields:
            filters_applied.append(f"minlength{min_length}")
        if balancing_filter_active and selected_balancing_fields:
            filters_applied.append("balancing")
        if duplicate_filter_active and duplicate_fields:
            filters_applied.append("duplicates")
        if remove_fields_filter_active and fields_to_remove:
            filters_applied.append("removefields")
        
        filters_suffix = '_'.join(filters_applied) if filters_applied else "filtered"
        selected_file = st.session_state.get('selected_file', 'output.json')
        base_name = os.path.splitext(selected_file)[0]
        output_filename = f"{base_name}_{filters_suffix}_{timestamp}.json"
        
        # Convert the filtered data to JSON bytes
        filtered_json_bytes = orjson.dumps(filtered_data, option=orjson.OPT_INDENT_2)
        
        # Provide a download button
        st.download_button(
            label="📥 Download Filtered JSON File",
            data=filtered_json_bytes,
            file_name=output_filename,
            mime="application/json"
        )
        st.write(f"Number of records after filtering: {len(filtered_data)}")
        st.success(f"Filtered data is available for download as: {output_filename}")

def data_viewer_tab(df):
    st.header("📁 Data Viewer")

    with st.expander("🔍 Record Preview"):
        if 'viewer_index' not in st.session_state:
            st.session_state.viewer_index = 0
        current_record = preview_data(df.to_dict(orient='records'), st.session_state.viewer_index)
        st.json(current_record)
        
        # Display current record number after button click
        st.write(f"**Record Number:** {st.session_state.viewer_index + 1}")

        # Buttons to navigate (now below the elements)
        col_prev, col_next = st.columns([1,1])
        with col_prev:
            if st.button("⬅️ Previous Record", key='prev_viewer'):
                if st.session_state.viewer_index > 0:
                    st.session_state.viewer_index -= 1
        with col_next:
            if st.button("➡️ Next Record", key='next_viewer'):
                if st.session_state.viewer_index < len(df) - 1:
                    st.session_state.viewer_index += 1

    st.subheader("📊 Display Records")
    st.markdown("**Note:** Large datasets may take a long time to display.")
    col_start, col_end = st.columns(2)
    with col_start:
        start_num = st.number_input("Start Record Number", min_value=1, value=1, step=1, key='start_num')
    with col_end:
        end_num = st.number_input("End Record Number", min_value=1, value=min(len(df), 10), step=1, key='end_num')
    st.write("")  # Blank line
    if st.button("🔄 Show Data"):
        if end_num < start_num:
            st.error("End number must be greater than or equal to start number.")
        elif end_num > len(df):
            st.error(f"End number cannot be greater than the number of records ({len(df)}).")
        else:
            st.write(df.iloc[start_num-1:end_num])

def werteverteilung_tab(df):
    st.header("📈 Value Distribution")
    metadata_fields = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if not metadata_fields:
        st.write("No metadata fields found.")
    else:
        search_field = st.text_input("Search Field", "", key='metadata_search')
        if search_field:
            filtered_fields = [field for field in metadata_fields if search_field.lower() in field.lower()]
        else:
            filtered_fields = metadata_fields
        if filtered_fields:
            selected_fields = st.multiselect("Select Metadata Fields for Visualization", filtered_fields, key='metadata_select_multi')
            if selected_fields:
                for field in selected_fields:
                    # Handle multiple values by splitting
                    value_series = df[field].dropna().astype(str).str.split(', ').explode()
                    value_counts = value_series.value_counts().head(20)
                    st.write(value_counts.to_frame().reset_index().rename(columns={'index': field, field: 'Count'}))
                    
                    # Truncate labels for readability
                    df_counts = value_counts.to_frame().reset_index()
                    df_counts.columns = [field, 'Count']
                    df_counts[field] = df_counts[field].apply(lambda x: x[:20] + '...' if len(x) > 20 else x)
                    
                    fig = px.bar(
                        df_counts,
                        x='Count',
                        y=field,
                        orientation='h',
                        labels={'Count': 'Number of Occurrences', field: 'Field'},
                        title="",  # Remove the title above the chart
                        hover_data={'Count': True}
                    )
                    fig.update_traces(marker_color='blue')
                    st.plotly_chart(fig, use_container_width=True, key=f'value_distribution_{field}')
        else:
            st.write("No fields found matching the search term.")

def fuellstandsanalyse_tab(df):
    st.header("📊 Fill Level Analysis")
    st.write("Filter data based on fields and values and analyze the fill level of metadata.")

    # Selection of fields and values for filtering
    st.subheader("🔍 Select Filters")
    selected_fill_fields = st.multiselect(
        "Select fields for filtering",
        options=df.columns.tolist(),
        default=[]
    )
    fill_field_values = {}
    for field in selected_fill_fields:
        unique_values = get_unique_values(st.session_state.original_json, field)
        selected_values = st.multiselect(f"Select values for {field}", unique_values, default=[], key=f"fill_{field}")
        fill_field_values[field] = selected_values

    # Option to apply filters jointly or separately
    join_option = st.radio(
        "How should the filters be applied?",
        options=["View Separately", "Create Combined Data Set"],
        index=0
    )

    # Selection of fields to display fill levels
    st.subheader("📈 Display Fill Levels for:")
    display_fill_fields = st.multiselect(
        "Select metadata fields to display fill levels",
        options=df.columns.tolist(),
        default=[]
    )

    if st.button("🔄 Analyze Fill Levels"):
        st.info("Starting fill level analysis...")
        if selected_fill_fields and any(fill_field_values[field] for field in selected_fill_fields):
            if join_option == "Create Combined Data Set":
                # Filter data that meets all selected field-value combinations
                filtered_df = df.copy()
                for field, values in fill_field_values.items():
                    if values:
                        filtered_df = filtered_df[filtered_df[field].isin(values)]
                subsets = {"Combined Data Set": filtered_df}
            else:
                # Each field-value combination as a separate subset
                subsets = {}
                for field, values in fill_field_values.items():
                    if values:
                        for value in values:
                            subset_name = f"{field} = {value}"
                            subsets[subset_name] = df[df[field].isin([value])]
        else:
            # No filters applied, a single subset
            subsets = {"All Data": df}

        if display_fill_fields:
            # Limit the number of charts to maintain clarity
            max_columns = 2  # Changed from 3 to 2 columns
            num_subsets = len(subsets)
            num_cols = min(max_columns, num_subsets)
            cols = st.columns(num_cols)
            
            for idx, (subset_name, subset_df) in enumerate(subsets.items()):
                col = cols[idx % num_cols]
                with col:
                    # Add a note above the chart
                    st.markdown(f"**Filter:** {subset_name}")
                    
                    # Remove the title above the charts
                    fill_status = subset_df[display_fill_fields].notnull().mean() * 100
                    fill_status = fill_status.sort_values(ascending=False)

                    # Dynamic adjustment of chart size
                    num_bars = len(fill_status)
                    if num_bars == 1:
                        fig_height = 400
                    else:
                        fig_height = max(400, num_bars * 50)
                    
                    # Prepare data for dual-colored bars
                    fill_percentage = fill_status
                    empty_percentage = 100 - fill_status
                    fill_data = pd.DataFrame({
                        'Metadata Field': fill_status.index,
                        'Filled (%)': fill_percentage.values,
                        'Empty (%)': empty_percentage.values
                    })

                    # Melt the DataFrame for stacked bar charts
                    fill_data_melted = fill_data.melt(id_vars='Metadata Field', value_vars=['Filled (%)', 'Empty (%)'], var_name='Status', value_name='Percentage')

                    fig = px.bar(
                        fill_data_melted,
                        x='Percentage',
                        y='Metadata Field',
                        color='Status',
                        orientation='h',
                        title="",  # Remove the title above the chart
                        labels={'Percentage': 'Percentage (%)', 'Metadata Field': 'Metadata Field'},
                        hover_data={'Percentage': True, 'Status': True}
                    )
                    fig.update_layout(barmode='stack', height=fig_height, showlegend=True)
                    st.plotly_chart(fig, use_container_width=True, key=f'fill_level_plot_{idx}')
        else:
            st.warning("Please select at least one field to display fill levels.")

def main():
    st.set_page_config(page_title="Universal JSON Data Tool", layout="wide")

    # Sidebar for title and JSON file selection
    with st.sidebar:
        st.title("Universal JSON Data Tool")
        st.write("This tool allows for the analysis and editing of JSON data with any structure.")
        
        uploaded_file = st.file_uploader("📂 Select a JSON file to upload", type=["json"], key='file_uploader')
        
        if uploaded_file:
            with st.spinner("Processing file..."):
                df = process_uploaded_file(uploaded_file)
                if not df.empty:
                    df = merge_similar_fields(df)
                    st.session_state['df'] = df

        # Footer in the sidebar
        st.markdown("---")
        st.markdown(
            """
            &copy; 2024 Jan Schachtschabel

            License: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)
            """
        )

    # Check if data is loaded
    if 'df' in st.session_state and not st.session_state['df'].empty:
        df = st.session_state['df']
        selected_file = st.session_state.get('selected_file', 'output.json')

        # Create tabs in the main area
        tabs = st.tabs(["Data Viewer", "Value Distribution", "Fill Level Analysis", "Text Analysis", "Data Filter"])
        
        with tabs[0]:
            data_viewer_tab(df)

        with tabs[1]:
            werteverteilung_tab(df)

        with tabs[2]:
            fuellstandsanalyse_tab(df)

        with tabs[3]:
            text_analysis_tab(df)

        with tabs[4]:
            json_filter_tab(df)
    else:
        st.info("Please upload a JSON file to begin analysis.")

if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


st.set_page_config(
    page_title="What Drives Medical Insurance Costs?",
    page_icon="💊",
    layout="wide",
)

sns.set_theme(style="darkgrid")


st.markdown(
    """
    <style>
    .stApp {
        background: #0f1117;
        color: #f5f5f7;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #075985 0%, #0f172a 100%);
    }

    [data-testid="stSidebar"] * {
        color: #ffffff;
    }

    h1, h2, h3 {
        color: #ffffff;
        letter-spacing: 0;
    }

    .section-block {
        padding: 1.1rem 0 0.5rem 0;
        border-top: 1px solid rgba(255, 255, 255, 0.12);
        margin-top: 1.2rem;
    }

    .insight-card {
        background: #e0f2fe;
        color: #0f2f44;
        border-left: 5px solid #0284c7;
        border-radius: 8px;
        padding: 1rem 1.1rem;
        min-height: 118px;
    }

    .insight-card h4 {
        color: #0f2f44;
        margin: 0 0 0.55rem 0;
    }

    .small-note {
        color: #c7c7d1;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    data = pd.read_csv("insurance.csv")
    data["charge_group"] = pd.cut(
        data["charges"],
        bins=[0, 10000, 25000, data["charges"].max() + 1],
        labels=["Low", "Medium", "High"],
    )
    return data


@st.cache_resource
def build_model(data):
    model_data = data.drop(columns=["charge_group"]).copy()

    features = ["age", "sex", "bmi", "children", "smoker", "region"]
    target = "charges"

    x = model_data[features]
    y = model_data[target]

    transformer = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                ["sex", "smoker", "region"],
            ),
            ("numeric", "passthrough", ["age", "bmi", "children"]),
        ]
    )

    model = Pipeline(
        steps=[
            ("transformer", transformer),
            ("linear_regression", LinearRegression()),
        ]
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    return {
        "model": model,
        "features": features,
        "x_test": x_test,
        "y_test": y_test,
        "predictions": predictions,
        "mae": metrics.mean_absolute_error(y_test, predictions),
        "rmse": np.sqrt(metrics.mean_squared_error(y_test, predictions)),
        "r2": metrics.r2_score(y_test, predictions),
    }


def dollars(value):
    return f"${value:,.2f}"


def page_header(title, subtitle):
    st.title(title)
    if subtitle:
        st.markdown(f"<p class='small-note'>{subtitle}</p>", unsafe_allow_html=True)


def divider():
    st.markdown("<div class='section-block'></div>", unsafe_allow_html=True)


def card(title, body):
    st.markdown(
        f"""
        <div class="insight-card">
            <h4>{title}</h4>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def predict_charge(model, age, sex, bmi, children, smoker, region):
    user_row = pd.DataFrame(
        {
            "age": [age],
            "sex": [sex],
            "bmi": [bmi],
            "children": [children],
            "smoker": [smoker],
            "region": [region],
        }
    )
    return model.predict(user_row)[0]


df = load_data()
model_info = build_model(df)


st.sidebar.markdown("## Midterm Project")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    [
        "Introduction Page",
        "Visualization Page",
        "Prediction Page",
        "Conclusion & Insights",
    ],
)
st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset")
st.sidebar.markdown("Medical Cost Personal Dataset")
st.sidebar.markdown("---")
st.sidebar.caption("What Drives Medical Insurance Costs?")


if page == "Introduction Page":
    page_header(
        "What Drives Medical Insurance Costs?",
        "",
    )
    st.markdown("### A predictive analysis of personal factors behind insurance charges")
    st.markdown(
        """
        <div style="
            height: 2px;
            width: 100%;
            background: linear-gradient(90deg, #38bdf8, #0ea5e9, #e0f2fe);
            margin: 0.4rem 0 1.2rem 0;
        "></div>
        """,
        unsafe_allow_html=True,
    )

    image_left, image_center, image_right = st.columns([1, 2, 1])
    with image_center:
        st.image("insurance.jpg", width="stretch")

    st.markdown("## Why This Topic")
    st.write(
        """
        Medical insurance costs are important to many people, but the reasons
        behind the price are not always easy to understand. This dataset shows
        how personal factors such as age, BMI, smoking status, children, sex,
        and region are related to insurance charges.

        The topic is useful because it connects data analysis with a real-life
        financial decision. Insurance charges can affect monthly budgets,
        healthcare access, and personal planning. By exploring this dataset, it
        becomes easier to see which factors are linked to higher costs and how
        different personal profiles can lead to different predicted charges.

        This topic is also practical because the variables are easy to understand.
        For example, age and BMI are health-related measurements, while smoking
        status reflects a lifestyle factor that can strongly affect expected
        medical risk. Because these variables are familiar, the results can be
        explained clearly in both the dashboard and the presentation.
        """
    )

    divider()

    st.markdown("## Project Goal")
    st.write(
        """
        The goal of this project is to identify the most influential factors
        behind medical insurance charges and use those factors to estimate an
        individual's expected insurance cost.

        - Explore how insurance charges are distributed across the dataset.
        - Compare insurance charges between smokers and non-smokers.
        - Analyze how age is related to insurance charges.
        - Predict insurance charges from personal information.
        - Summarize the main insights in the conclusion page.
        """
    )

    divider()

    st.markdown("## Dataset at a Glance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{df.shape[0]:,}")
    m2.metric("Features", "6")
    m3.metric("Target", "charges")
    m4.metric("Average Charge", dollars(df["charges"].mean()))

    preview_tab, dictionary_tab = st.tabs(["First Rows", "Feature Dictionary"])

    with preview_tab:
        st.dataframe(df.drop(columns=["charge_group"]).head(10), width="stretch")

    with dictionary_tab:
        dictionary = pd.DataFrame(
            {
                "Feature": [
                    "age",
                    "sex",
                    "bmi",
                    "children",
                    "smoker",
                    "region",
                    "charges",
                ],
                "Description": [
                    "Age of the insurance beneficiary",
                    "Sex of the beneficiary",
                    "Body Mass Index",
                    "Number of children covered by insurance",
                    "Smoking status",
                    "Residential region in the United States",
                    "Medical insurance cost, used as the prediction target",
                ],
                "Role": [
                    "Input",
                    "Input",
                    "Input",
                    "Input",
                    "Input",
                    "Input",
                    "Target",
                ],
            }
        )
        st.dataframe(dictionary, width="stretch", hide_index=True)

    divider()

    st.markdown("## Descriptive Statistics")
    st.dataframe(df.drop(columns=["charge_group"]).describe().T, width="stretch")


elif page == "Visualization Page":
    page_header(
        "Data Visualization",
        "Interactive charts that explain the most important patterns in the insurance data.",
    )

    st.markdown("## Choose a Question to Explore")
    question = st.selectbox(
        "Visualization question",
        [
            "How are charges distributed?",
            "How much does smoking affect charges?",
            "How does age affect charges?",
            "Do regions have different average charges?",
        ],
    )

    if question == "How are charges distributed?":
        fig, ax = plt.subplots(figsize=(11, 5))
        sns.histplot(df["charges"], bins=35, kde=True, color="#0ea5e9", ax=ax)
        ax.set_title("Distribution of Medical Insurance Charges")
        ax.set_xlabel("Insurance Charges")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        card(
            "Insight",
            "The distribution is right-skewed. Most people have relatively lower charges, while a smaller group has very high charges.",
        )

    elif question == "How much does smoking affect charges?":
        left, right = st.columns([1.2, 1])
        with left:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=df, x="smoker", y="charges", palette="Blues", ax=ax)
            ax.set_title("Charges by Smoking Status")
            ax.set_xlabel("Smoker")
            ax.set_ylabel("Insurance Charges")
            st.pyplot(fig)
        with right:
            smoker_summary = df.groupby("smoker", as_index=False)["charges"].mean()
            smoker_summary["average_charges"] = smoker_summary["charges"].map(dollars)
            st.dataframe(
                smoker_summary[["smoker", "average_charges"]],
                width="stretch",
                hide_index=True,
            )
            smoker_gap = (
                df.loc[df["smoker"] == "yes", "charges"].mean()
                - df.loc[df["smoker"] == "no", "charges"].mean()
            )
            card(
                "Insight",
                f"Smokers pay about {dollars(smoker_gap)} more on average than non-smokers in this dataset.",
            )

    elif question == "How does age affect charges?":
        fig, ax = plt.subplots(figsize=(11, 5))
        sns.scatterplot(
            data=df,
            x="age",
            y="charges",
            hue="smoker",
            alpha=0.65,
            palette={"yes": "#ef4444", "no": "#38bdf8"},
            ax=ax,
        )
        ax.set_title("Age and Insurance Charges")
        ax.set_xlabel("Age")
        ax.set_ylabel("Insurance Charges")
        st.pyplot(fig)
        card(
            "Insight",
            "Insurance charges generally increase as age increases.",
        )

    elif question == "Do regions have different average charges?":
        region_summary = (
            df.groupby("region", as_index=False)["charges"]
            .mean()
            .sort_values("charges", ascending=False)
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=region_summary, x="region", y="charges", color="#0ea5e9", ax=ax)
        ax.set_title("Average Insurance Charges by Region")
        ax.set_xlabel("Region")
        ax.set_ylabel("Average Insurance Charges")
        st.pyplot(fig)
        card(
            "Insight",
            "Regional differences exist, but the difference is smaller than the effect of smoking status.",
        )


elif page == "Prediction Page":
    page_header(
        "Model Prediction",
        "A Linear Regression model predicts medical insurance charges from personal information.",
    )

    st.markdown("## Prediction Simulator")
    form_left, form_right = st.columns(2)

    with form_left:
        age = st.slider("Age", 18, 64, 35)
        bmi = st.slider("BMI", 15.0, 55.0, 28.0, 0.1)
        children = st.number_input("Children", min_value=0, max_value=5, value=0)

    with form_right:
        sex = st.selectbox("Sex", sorted(df["sex"].unique()))
        smoker = st.selectbox("Smoker", sorted(df["smoker"].unique()))
        region = st.selectbox("Region", sorted(df["region"].unique()))

    predicted_charge = predict_charge(
        model_info["model"],
        age,
        sex,
        bmi,
        children,
        smoker,
        region,
    )

    st.markdown("## Predicted Insurance Charge")
    st.metric("Estimated Cost", dollars(predicted_charge))

    divider()

    st.markdown("## Actual vs Predicted")
    fig, ax = plt.subplots(figsize=(9, 5))
    y_test = model_info["y_test"]
    predictions = model_info["predictions"]
    ax.scatter(y_test, predictions, alpha=0.65, color="#0ea5e9")
    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color="#fb7185",
        linestyle="--",
        linewidth=2,
    )
    ax.set_title("Actual Charges vs Predicted Charges")
    ax.set_xlabel("Actual Charges")
    ax.set_ylabel("Predicted Charges")
    st.pyplot(fig)
    st.caption("The dashed line shows perfect prediction. Points closer to the line are more accurate.")


elif page == "Conclusion & Insights":
    smoker_avg = df.groupby("smoker")["charges"].mean()
    smoker_gap = smoker_avg["yes"] - smoker_avg["no"]
    age_corr = df["age"].corr(df["charges"])
    bmi_corr = df["bmi"].corr(df["charges"])

    page_header(
        "Conclusion and Insights",
        "",
    )

    st.markdown("## Key Takeaways")
    a, b, c = st.columns(3)
    with a:
        card(
            "Smoking Is the Strongest Driver",
            f"Smokers have much higher average charges. The average gap is {dollars(smoker_gap)}.",
        )
    with b:
        card(
            "Age Matters",
            f"Age has a positive relationship with charges. The correlation is {age_corr:.2f}.",
        )
    with c:
        card(
            "BMI Adds Risk",
            f"BMI is also related to insurance cost. The correlation is {bmi_corr:.2f}.",
        )

    divider()

    st.markdown("## Overall Conclusion")
    st.write(
        f"""
        Based on the overall data analysis, medical insurance charges are not
        randomly distributed. Personal factors such as smoking status, age, and
        BMI show clear relationships with insurance costs. Among these variables,
        smoking status appears to be the strongest cost driver. In this dataset,
        smokers pay about **{dollars(smoker_gap)}** more on average than
        non-smokers.

        Age also has a positive relationship with insurance charges. The
        correlation between age and charges is **{age_corr:.2f}**, which means
        charges tend to increase as people get older. BMI also shows a positive
        relationship with charges, although its effect is weaker than the
        difference between smokers and non-smokers.
        """
    )

    st.markdown("## Health and Insurance Advisory")
    st.write(
        """
        The most important insight from this project is that lifestyle and health
        indicators can influence expected insurance costs. Smoking is especially
        important because it is connected to higher medical risk and much higher
        predicted insurance charges. Reducing smoking-related risk can be a
        meaningful long-term health and financial decision.

        Maintaining a healthier BMI may also help reduce health risks over time.
        While the model does not prove that BMI directly causes higher insurance
        charges, the data shows that higher BMI values are associated with higher
        costs in many cases.
        """
    )

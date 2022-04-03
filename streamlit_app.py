import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import random


VARIABLES = {
    "a": {"degree": "x²", "limits": (-2.000, 2.000, -0.450)},
    "b": {"degree": "x", "limits": (-1.000, 1.000, -0.200)},
    "c": {"degree": "", "limits": (0.000, 1.000, 1.000)},
}

GRAPH = {
    "font": {"size": 30, "color": "white"},
    "gridcolor": "rgba(64,64,64,255)",
    "plot_bgcolor": "rgba(32,32,32,255)",
    "margin": {"l": 20, "r": 0, "t": 50, "b": 20},
    "height": 400,
    "width": 1000,
}

WEAPONS = ["PDW", "SMG", "Carbine", "AR", "BR"]


def parse_equation(coefficients: list[float]) -> str:
    equation = f""

    for coefficient, variable in zip(coefficients, VARIABLES.values()):
        if coefficient >= 0:
            equation += f"+{coefficient}{variable['degree']}"
        else:
            equation += f"{coefficient}{variable['degree']}"

    return equation


if __name__ == "__main__":
    st.set_page_config(
        page_title="Parameters Calculator",
        page_icon="🎮",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    df = pd.DataFrame()

    st.sidebar.title("Parameters")
    data_points = st.sidebar.slider("Data Points", 2, 100, 50)

    st.title("Modifiers")

    # ACCURACY

    container = st.container()
    cols = st.columns([1, 20, 20, 20, 1])
    coefficients = []

    for i, const in enumerate(VARIABLES.keys()):
        with cols[i + 1]:
            coefficients.append(
                st.slider(f"Accuracy: {const}", *VARIABLES[const]["limits"])
            )

    equation_accuracy = parse_equation(coefficients)

    x = np.linspace(0, 1, data_points)
    y_accuracy = np.clip(
        coefficients[0] * x ** 2 + coefficients[1] * x + coefficients[2], 0, 1
    )

    df["x"] = x
    df["modifier"] = y_accuracy

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_accuracy,
            name="Accuracy Modifier",
            line={"color": "royalblue", "width": 4},
        )
    )

    fig.update_layout(
        title={"text": "Accuracy Modifier", "font": GRAPH["font"]},
        xaxis={"gridcolor": GRAPH["gridcolor"], "range": (0, 1)},
        yaxis={"gridcolor": GRAPH["gridcolor"], "range": (0, 1)},
        showlegend=False,
        height=GRAPH["height"],
        width=GRAPH["width"],
        margin=GRAPH["margin"],
        plot_bgcolor=GRAPH["plot_bgcolor"],
    )

    # fig.add_annotation(x=0.1, y=10, text=equation_accuracy, font={"size": 20}, showarrow=False)

    with container:
        st.plotly_chart(fig)

    # CROUCH AND PRONE

    container = st.container()
    cols = st.columns([1, 20, 20, 20, 1])
    coefficients = {"crouch": [], "prone": []}

    for i, const in enumerate(VARIABLES.keys()):
        with cols[i + 1]:
            coefficients["crouch"].append(
                st.slider(f"Crouch: {const}", *VARIABLES[const]["limits"])
            )
            coefficients["prone"].append(
                st.slider(f"Prone: {const}", *VARIABLES[const]["limits"])
            )

    equation_crouch = parse_equation(coefficients["crouch"])
    equation_prone = parse_equation(coefficients["prone"])

    y_crouch = np.clip(
        coefficients["crouch"][0] * x ** 2
        + coefficients["crouch"][1] * x
        + coefficients["crouch"][2],
        0,
        1,
    )
    y_prone = np.clip(
        coefficients["prone"][0] * x ** 2
        + coefficients["prone"][1] * x
        + coefficients["prone"][2],
        0,
        1,
    )

    df["crouch_0"] = y_crouch
    df["prone_0"] = y_prone

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=y_crouch, name="Crouch", line={"color": "firebrick", "width": 4}
        )
    )
    fig.add_trace(
        go.Scatter(x=x, y=y_prone, name="Prone", line={"color": "orange", "width": 4})
    )

    fig.update_layout(
        title={"text": "Baselines", "font": GRAPH["font"]},
        xaxis={"gridcolor": GRAPH["gridcolor"], "range": (0, 1)},
        yaxis={"gridcolor": GRAPH["gridcolor"], "range": (0, 1)},
        showlegend=False,
        height=GRAPH["height"],
        width=GRAPH["width"],
        margin=GRAPH["margin"],
        plot_bgcolor=GRAPH["plot_bgcolor"],
    )

    with container:
        st.plotly_chart(fig)

    # Multiplied

    y_crouch *= y_accuracy
    y_prone *= y_accuracy

    df["crouch_1"] = y_crouch
    df["prone_1"] = y_prone

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=y_crouch, name="Crouch", line={"color": "firebrick", "width": 4}
        )
    )
    fig.add_trace(
        go.Scatter(x=x, y=y_prone, name="Prone", line={"color": "orange", "width": 4})
    )

    fig.update_layout(
        title={"text": "Multiplied Baselines", "font": GRAPH["font"]},
        xaxis={"gridcolor": GRAPH["gridcolor"], "range": (0, 1)},
        yaxis={"gridcolor": GRAPH["gridcolor"], "range": (0, 1)},
        showlegend=False,
        height=GRAPH["height"],
        width=GRAPH["width"],
        margin=GRAPH["margin"],
        plot_bgcolor=GRAPH["plot_bgcolor"],
    )

    st.plotly_chart(fig)

    # Weapons

    st.title("Weapons")
    st.subheader("Accuracy")

    cols = st.columns([1.7, 20, 20, 20, 20, 20, 1.7])

    weapon_accuracy_crouch = {}
    weapon_accuracy_prone = {}

    fig = go.Figure()

    for i, weapon in enumerate(WEAPONS):
        with cols[i + 1]:
            acc = st.slider(f"{weapon}", 0.000, 1.000, 0.500)

            weapon_accuracy_crouch[weapon] = acc * y_crouch
            weapon_accuracy_prone[weapon] = acc * y_prone

    for weapon in WEAPONS:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=weapon_accuracy_crouch[weapon],
                name=weapon,
                line={"width": 4},
            )
        )

    fig.update_layout(
        title={"text": "Crouch", "font": GRAPH["font"]},
        xaxis={"gridcolor": GRAPH["gridcolor"], "range": (0, 1)},
        yaxis={"gridcolor": GRAPH["gridcolor"], "range": (0, 1)},
        height=GRAPH["height"],
        width=GRAPH["width"],
        margin=GRAPH["margin"],
        plot_bgcolor=GRAPH["plot_bgcolor"],
    )

    st.plotly_chart(fig)

    fig = go.Figure()

    for weapon in WEAPONS:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=weapon_accuracy_prone[weapon],
                name=weapon,
                line={"width": 4},
            )
        )

    fig.update_layout(
        title={"text": "Prone", "font": GRAPH["font"]},
        xaxis={"gridcolor": GRAPH["gridcolor"], "range": (0, 1)},
        yaxis={"gridcolor": GRAPH["gridcolor"], "range": (0, 1)},
        height=GRAPH["height"],
        width=GRAPH["width"],
        margin=GRAPH["margin"],
        plot_bgcolor=GRAPH["plot_bgcolor"],
    )

    st.plotly_chart(fig)

    df["crouch_1"] = y_crouch
    df["prone_1"] = y_prone

    for weapon in WEAPONS:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=weapon_accuracy_crouch[weapon],
                name=weapon,
                line={"width": 4},
            )
        )

    for weapon in WEAPONS:
        df[f"{weapon}_crouch"] = weapon_accuracy_crouch[weapon]
        df[f"{weapon}_prone"] = weapon_accuracy_prone[weapon]

    # Download

    st.sidebar.title("Export Data")
    st.sidebar.download_button(
        "Download CSV", df.to_csv(index=False), file_name="weapon_accuracy_curves.csv"
    )

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.ensemble import RandomForestClassifier

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Pipeline de Preprocesamiento Interactivo")

archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])
if archivo is not None:
    df = pd.read_csv(archivo)
    st.subheader("📊 Estadística descriptiva")
    st.write(df.describe())

    st.subheader("🔍 Valores nulos por columna")
    st.write(df.isnull().sum())

    st.subheader("📦 Outliers por IQR")
    outlier_summary = []
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower) | (df[col] > upper)].shape[0]
        outlier_summary.append({"columna": col, "outliers": outliers})
    st.write(pd.DataFrame(outlier_summary))

    st.subheader("📈 Boxplots de las primeras 6 variables numéricas")
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns[:6]
    for col in num_cols:
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(f"Boxplot de {col}")
        st.pyplot(fig)

    st.subheader("🌡️ Matriz de correlación")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.select_dtypes(include=['float64', 'int64']).corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.subheader("🧼 Imputación de valores nulos")
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            metodo = st.selectbox(f"Imputación para '{col}' ({df[col].isnull().sum()} nulos):",
                                  ['No imputar', 'media', 'mediana', 'moda'], key=col)
            if metodo == 'media':
                df[col].fillna(df[col].mean(), inplace=True)
            elif metodo == 'mediana':
                df[col].fillna(df[col].median(), inplace=True)
            elif metodo == 'moda':
                df[col].fillna(df[col].mode()[0], inplace=True)

    st.subheader("📏 Escalado")
    cols_numericas = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    columnas_a_escalar = st.multiselect("Selecciona columnas para escalar", cols_numericas)
    if columnas_a_escalar:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df[columnas_a_escalar])
        for i, col in enumerate(columnas_a_escalar):
            df[col + '_scaled'] = scaled[:, i]
        st.success("Variables escaladas agregadas.")

    st.subheader("🏷 Conversión a dummies")
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    cols_dummies = st.multiselect("Selecciona columnas para convertir a dummies", cat_cols)
    if cols_dummies:
        df = pd.get_dummies(df, columns=cols_dummies, drop_first=True, dtype=int)
        st.success("Dummies creadas.")

    st.subheader("🧠 Selección de características")
    target = st.selectbox("Selecciona la variable objetivo (target)", df.columns)
    metodo = st.selectbox("Método de selección", ['correlacion', 'chi2', 'rfe'])
    k = st.slider("Número de características a seleccionar", 2, min(10, df.shape[1]), 5)

    def seleccionar_caracteristicas(df, metodo, k, target):
        df_temp = df.copy()
        df_temp = df_temp.select_dtypes(exclude='object')
        if metodo == 'correlacion':
            corr_matrix = df_temp.corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            columnas_a_eliminar = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
            st.write("Variables eliminadas por alta correlación:", columnas_a_eliminar)
            return df.drop(columns=columnas_a_eliminar)
        elif metodo == 'chi2':
            X = df_temp.drop(columns=[target])
            y = df_temp[target]
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            selector = SelectKBest(score_func=chi2, k=k)
            selector.fit(X_scaled, y)
            selected = X.columns[selector.get_support()]
            st.write("Variables seleccionadas por chi2:", selected.tolist())
            return df[selected.tolist() + [target]]
        elif metodo == 'rfe':
            X = df_temp.drop(columns=[target])
            y = df_temp[target]
            model = RandomForestClassifier(random_state=42)
            selector = RFE(estimator=model, n_features_to_select=k)
            selector.fit(X, y)
            selected = X.columns[selector.support_]
            st.write("Variables seleccionadas por RFE:", selected.tolist())
            return df[selected.tolist() + [target]]
        else:
            st.warning("Método no reconocido. Se retorna el DataFrame completo.")
            return df

    df_final = seleccionar_caracteristicas(df, metodo, k, target)

    st.subheader("🔥 Matriz de correlación final")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df_final.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.success("✅ Preprocesamiento completado. Aquí tienes tu DataFrame listo para modelar.")
    st.dataframe(df_final.head())


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout='wide')
st.title("🔧 Dashboard Interactivo de Preprocesamiento - Olist Dataset")

# Cargar archivo
uploaded_file = st.sidebar.file_uploader("📁 Sube tu archivo CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Archivo cargado correctamente.")
    
    st.subheader("Vista previa")
    st.dataframe(df.head())

    # Paso 1: Estadística Descriptiva
    st.markdown("### Estadística descriptiva")
    st.dataframe(df.describe())

    # Paso 2: Valores nulos
    st.markdown("### Valores nulos por columna")
    nulls = df.isnull().sum()
    st.dataframe(nulls[nulls > 0])

    # Imputación
    st.markdown("### Imputación de valores nulos")
    cols_with_nulls = df.columns[df.isnull().sum() > 0].tolist()
    for col in cols_with_nulls:
        method = st.selectbox(f"Imputar '{col}' con:", ["No imputar", "Media", "Mediana", "Moda"], key=col)
        if method == "Media":
            df[col].fillna(df[col].mean(), inplace=True)
        elif method == "Mediana":
            df[col].fillna(df[col].median(), inplace=True)
        elif method == "Moda":
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Paso 3: Outliers
    st.markdown("### Outliers (IQR)")
    outlier_data = []
    for col in df.select_dtypes(include=['float64', 'int64']):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
        outlier_data.append({"Columna": col, "Outliers": len(outliers)})
    st.dataframe(pd.DataFrame(outlier_data))

    # Boxplots
    st.markdown("### Boxplots de variables numéricas")
    box_cols = st.multiselect("Selecciona columnas para boxplot", df.select_dtypes(include=['float64', 'int64']).columns)
    for col in box_cols:
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        st.pyplot(fig)

    # Transformaciones
    st.markdown("### Transformaciones")
    transform_cols = st.multiselect("Selecciona columnas a transformar (log/sqrt)", df.select_dtypes(include=['float64', 'int64']).columns)
    tipo = st.radio("Tipo de transformación:", ["Ninguna", "Log", "Raíz cuadrada", "Ambas"])
    if tipo != "Ninguna":
        for col in transform_cols:
            if tipo in ["Log", "Ambas"]:
                df[col + "_log"] = np.log1p(df[col])
            if tipo in ["Raíz cuadrada", "Ambas"]:
                df[col + "_sqrt"] = np.sqrt(df[col])
        st.success("✔️ Transformaciones aplicadas.")

    # Ingeniería de características
    st.markdown("### Ingeniería de características")
    nueva_var = st.text_input("Agrega código para nueva variable (ej. df['nueva'] = df['col1'] * df['col2'])")
    if st.button("Ejecutar código"):
        try:
            exec(nueva_var)
            st.success("✔️ Variable agregada.")
        except Exception as e:
            st.error(f"❌ Error: {e}")

    # Escalado
    st.markdown("### Escalado de variables")
    cols_escalar = st.multiselect("Selecciona columnas para escalar", df.select_dtypes(include=['float64', 'int64']).columns)
    if st.checkbox("Aplicar escalado"):
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df[cols_escalar])
        for i, col in enumerate(cols_escalar):
            df[col + "_scaled"] = scaled[:, i]
        st.success("✔️ Escalado aplicado.")

    # Dummies
    st.markdown("### Conversión a dummies")
    categ_cols = df.select_dtypes(include='object').columns.tolist()
    cols_dummies = st.multiselect("Selecciona columnas categóricas para dummies", categ_cols)
    if cols_dummies:
        df = pd.get_dummies(df, columns=cols_dummies, drop_first=True, dtype=int)
        st.success("✔️ Dummies generadas.")

    # Selección de características
    st.markdown("### Selección de características")
    metodo = st.selectbox("Método de selección", ['correlacion', 'chi2', 'rfe'])
    target = st.selectbox("Selecciona la variable objetivo (target)", df.columns)
    k = st.slider("¿Cuántas características deseas seleccionar?", 1, 20, 10)

    def seleccionar_caracteristicas(df, metodo, k, target):
        df_temp = df.select_dtypes(exclude='object').copy()
        if metodo == 'correlacion':
            corr_matrix = df_temp.corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.9)]
            st.write("Variables eliminadas por alta correlación:", to_drop)
            return df.drop(columns=to_drop)
        elif metodo == 'chi2':
            X = df_temp.drop(columns=[target])
            y = df_temp[target]
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            selector = SelectKBest(score_func=chi2, k=k)
            selector.fit(X_scaled, y)
            selected = X.columns[selector.get_support()]
            st.write("Variables seleccionadas:", selected.tolist())
            return df[selected.tolist() + [target]]
        elif metodo == 'rfe':
            X = df_temp.drop(columns=[target])
            y = df_temp[target]
            model = RandomForestClassifier(random_state=42)
            selector = RFE(model, n_features_to_select=k)
            selector.fit(X, y)
            selected = X.columns[selector.support_]
            st.write("Variables seleccionadas:", selected.tolist())
            return df[selected.tolist() + [target]]
        return df

    if st.button("Aplicar selección de características"):
        df = seleccionar_caracteristicas(df, metodo, k, target)

    # Matriz final
    st.markdown("### Correlación final")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.select_dtypes(include=['float64', 'int64']).corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Exportar
    st.markdown("### Descargar DataFrame final")
    st.download_button("Descargar CSV", df.to_csv(index=False), "df_final.csv", "text/csv")

else:
    st.info("Esperando archivo CSV...")




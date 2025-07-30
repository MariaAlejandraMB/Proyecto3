def pipeline_premodelado_interactivo(df, target='target'):

    print("\n📊 Paso 1: Estadística descriptiva")
    print(df.describe())

    print("\n🔍 Valores nulos por columna:")
    print(df.isnull().sum())

    print("\n📦 Tabla de valores atípicos por IQR:")
    outlier_summary = []
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower) | (df[col] > upper)].shape[0]
        outlier_summary.append({"columna": col, "outliers": outliers})
    print(pd.DataFrame(outlier_summary))

    print("\n📈 Boxplots de variables numéricas:")
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns[:6]
    for col in num_cols:
        plt.figure(figsize=(6, 3))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot de {col}")
        plt.show()

    print("\n🌡️ Matriz de correlación:")
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.select_dtypes(include=['float64', 'int64']).corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Matriz de correlación")
    plt.show()

    print("\n🧼 Paso 2: Tratamiento de valores nulos (columna por columna)")
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            print(f"\n➡️ La columna '{col}' tiene {df[col].isnull().sum()} valores nulos.")
            decision = input(f"¿Deseas imputarla? (si/no): ").strip().lower()
            if decision == 'si':
                metodo = input("¿Método de imputación? (media/mediana/moda): ").strip().lower()
                if metodo == 'media':
                    df[col].fillna(df[col].mean(), inplace=True)
                elif metodo == 'mediana':
                    df[col].fillna(df[col].median(), inplace=True)
                elif metodo == 'moda':
                    df[col].fillna(df[col].mode()[0], inplace=True)
                else:
                    print(f"⚠️ Método '{metodo}' no reconocido. Se omite imputación para esta columna.")
            else:
                print(f"❗ No se imputó '{col}'.")

    print("\n🔁 Paso 3: Transformación de variables")
    aplicar_transform = input("¿Deseas aplicar transformaciones logarítmicas/raíz cuadrada? (si/no): ").strip().lower()
    if aplicar_transform == 'si':
        columnas_transformables = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        print(f"Variables disponibles para transformar: {columnas_transformables}")
        columnas_elegidas = input("Escribe las columnas a transformar separadas por coma: ").split(',')
        columnas_elegidas = [col.strip() for col in columnas_elegidas if col.strip() in columnas_transformables]

        tipo = input("¿Qué tipo de transformación deseas aplicar? (log/raiz/ambas): ").strip().lower()

        for col in columnas_elegidas:
            if tipo in ['log', 'ambas']:
                df[col + '_log'] = np.log1p(df[col])
                plt.figure(figsize=(6, 3))
                sns.boxplot(x=df[col + '_log'])
                plt.title(f"Boxplot de {col}_log")
                plt.show()
            if tipo in ['raiz', 'ambas']:
                df[col + '_sqrt'] = np.sqrt(df[col])
                plt.figure(figsize=(6, 3))
                sns.boxplot(x=df[col + '_sqrt'])
                plt.title(f"Boxplot de {col}_sqrt")
                plt.show()
        print("✔️ Transformaciones aplicadas y agregadas como nuevas columnas.")
    else:
        print("❗ No se aplicaron transformaciones.")

    print("\n🛠 Paso 4: Ingeniería de características")
    agregar_otra = input("¿Deseas agregar nuevas variables manualmente? (si/no): ").strip().lower()
    while agregar_otra == 'si':
        print("Escribe el código para crear una nueva variable (usa 'df' como referencia):")
        try:
            codigo = input(">>> ")
            exec(codigo)
            print("✔️ Variable agregada correctamente.")
        except Exception as e:
            print(f"❌ Error al crear la variable: {e}")
        agregar_otra = input("¿Deseas agregar otra variable? (si/no): ").strip().lower()

    print("\n📏 Paso 5: Escalado")
    escalar = input("¿Deseas escalar variables numéricas? (si/no): ").strip().lower()
    if escalar == 'si':
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        print(f"Variables numéricas disponibles: {numeric_cols}")
        cols_escalar = input("¿Qué columnas deseas escalar? (separadas por coma): ").split(',')
        cols_escalar = [col.strip() for col in cols_escalar if col.strip() in numeric_cols]
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df[cols_escalar])
        for i, col in enumerate(cols_escalar):
            df[col + '_scaled'] = scaled[:, i]
        print("✔️ Variables escaladas agregadas.")
    else:
        print("❗ No se aplicó escalado.")

    print("\n🏷 Paso 6: Conversión a variables dummies")
    categoricas = df.select_dtypes(include='object').columns.tolist()
    print(f"Variables categóricas disponibles: {categoricas}")
    columnas_dummies = input("¿Qué columnas deseas convertir en dummies? (separadas por coma): ").split(',')
    columnas_dummies = [col.strip() for col in columnas_dummies if col.strip() in categoricas]
    df = pd.get_dummies(df, columns=columnas_dummies, drop_first=True, dtype=int)
    print("✔️ Dummies agregadas al DataFrame.")

    print("\n🧠 Paso 7: Selección de características")
    print("Métodos disponibles: 'correlacion', 'chi2', 'rfe', 'rfe2'")
    metodo = input("¿Qué método de selección deseas aplicar?: ").strip().lower()
    try:
        k = int(input("¿Cuántas características deseas seleccionar?: "))
    except:
        print("Valor inválido para k. Se usarán 10 por defecto.")
        k = 10

    def seleccionar_caracteristicas(df, metodo, k=10, target='target'):
        df_temp = df.copy()
        df_temp = df_temp.select_dtypes(exclude='object')
        if metodo == 'correlacion':
            corr_matrix = df_temp.corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            columnas_a_eliminar = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
            print("Variables eliminadas por alta correlación:", columnas_a_eliminar)
            return df.drop(columns=columnas_a_eliminar)
        elif metodo == 'chi2':
            X = df_temp.drop(columns=[target])
            y = df_temp[target]
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            selector = SelectKBest(score_func=chi2, k=k)
            selector.fit(X_scaled, y)
            selected = X.columns[selector.get_support()]
            print("Variables seleccionadas por chi2:", selected.tolist())
            return df[selected.tolist() + [target]]
        elif metodo in ['rfe', 'rfe2']:
            X = df_temp.drop(columns=[target])
            y = df_temp[target]
            model = RandomForestClassifier(random_state=42)
            selector = RFE(estimator=model, n_features_to_select=k)
            selector.fit(X, y)
            selected = X.columns[selector.support_]
            print("Variables seleccionadas por RFE:", selected.tolist())
            return df[selected.tolist() + [target]]
        else:
            print("Método no reconocido. Se retorna el DataFrame completo.")
            return df

    df_final = seleccionar_caracteristicas(df, metodo=metodo, k=k, target=target)

    print("\n🔥 Matriz de correlación del DataFrame FINAL (con todas las variables antes de selección):")
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.select_dtypes(include=['float64', 'int64']).corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Matriz de correlación - DataFrame completo")
    plt.show()

    print("\n✅ Pipeline completado. DataFrame listo para modelar.")
    return df_final

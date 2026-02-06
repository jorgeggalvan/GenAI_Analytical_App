
# Importación de librerías
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from ydata_profiling import ProfileReport
import streamlit.components.v1 as components

import pandasql as ps

from google import genai

# Configuración inicial de la app de Streamlit
st.set_page_config(page_title = "GenAI App", 
                  page_icon="https://www.isdi.education/es/wp-content/uploads/2024/02/cropped-Favicon.png")

st.title("Sistema de Analítica Conversacional con IA Generativa") # Título principal de la app
st.divider() # Línea separadora

# %%
# ================================================================================
# 1 - SISTEMA DESCRIPTIVO ESTÁTICO
# ================================================================================

st.header("📊 N1. Sistema de Decisión Estático", divider='gray') # Encabezado de sección

# %%
# ================================================================================
# SECCIÓN 1.1 - CARGA DE DATOS
# ================================================================================

st.subheader("Datos disponibles") # Subtítulo

# Ruta del dataset
DATASET = "./data/adidas_sales.csv"

# Mostrar dataset interactivo básico
try:
    df = pd.read_csv(DATASET) # Lectura de dataset

    NUM_COL = "Units Sold" # Definición de columna numérica
    
    # Configuración de columna con barra de progreso
    COLUMN_CONFIG = {
        NUM_COL: st.column_config.ProgressColumn(
            min_value=0,                 # Valor mínimo de columna
            max_value=df[NUM_COL].max(), # Valor máximo de columna
            color="auto",                # Color automático según valor
            format="compact"             # Formato compacto de número
        )
    }

    st.write("A continuación se muestran los datos cargados:") # Texto explicativo
    #st.dataframe(df, column_config=COLUMN_CONFIG) # Mostrar tabla de datos

except Exception:
    st.error(f"No se encontró el archivo '{DATASET}'") # Mostrar error si no existe el dataset

# %%
# ================================================================================
# SECCIÓN DE FILTROS
# ================================================================================

# Cálculo de último mes y año
last_month = df['Month'].max()
last_year = df['Year'].max()

# Creacion de barra lateral
with st.sidebar:

    st.subheader("Filtros de datos") # Subtítulo

    # Filtros en barra lateral
    
    # Filtro de año
    years = st.sidebar.selectbox("Año", options=sorted(df['Year'].unique()), index=sorted(df['Year'].unique()).index(last_year))
    #Filtro de mes
    months = st.slider("Rango de meses", min_value=int(df['Month'].min()), max_value=int(df['Month'].max()), value=(last_month, last_month))
    # Filtro de métodos de venta
    methods = st.multiselect("Canal de venta", options=df["Sales Method"].unique(), default=df["Sales Method"].unique())
    # Filtro de retailer
    retailers = st.multiselect("Retailers", df["Retailer"].unique())
    # Filtro de productos
    products = st.multiselect("Productos", df["Product"].unique())

# Copia de DataFrame original
df_filtered = df.copy()

# Filtrar DataFrame por año
df_filtered = df_filtered[df_filtered['Year'] == years]

# Filtrar DataFrame por rango de meses
df_filtered = df_filtered[(df_filtered['Month'] >= months[0]) & (df_filtered['Month'] <= months[1])]

# Filtrar DataFrame por métodos de venta
if methods:
    df_filtered = df_filtered[df_filtered['Sales Method'].isin(methods)]

# Filtrar DataFrame por retailers
if retailers:
    df_filtered = df_filtered[df_filtered['Retailer'].isin(retailers)]

# Filtrar DataFrame por productos
if products:
    df_filtered = df_filtered[df_filtered['Product'].isin(products)]

# %%
# ================================================================================
# SECCIÓN 1.2 - CARGA DE DATOS AVANZADA
# ================================================================================

# Mostrar dataset interactivo avanzado
try:
    df_edited = st.data_editor(
        
        df_filtered,              # Dataset                       
        use_container_width=True, # Máximo ancho disponible          
        hide_index=True,          # Ocultar el índice
        num_rows="dynamic",       # Permitir agregar/eliminar filas
        disabled=[                # Columnas no editables
            'Year', 'Month', 'Year-Week', 'USA Region', 'Sales Method',
            'Price per Unit', 'Units Sold', 'Total Sales', 'Operating Profit', 'Operating Margin'
        ] 
    )

    df_filtered = df_edited # Actualización de DataFrame con los cambios realizados en el editor

except Exception as e:
    st.error(f"Error al mostrar los datos: {e}") # Mostrar mensaje de error

# %%
# ================================================================================
# SECCIÓN 1.3 - REPORTE DE PERFILADO
# ================================================================================

st.subheader("Reporte de perfilado") # Subtítulo

# Reporte de perfilado del DataFrame
with st.expander("Reporte de perfilado", expanded=False): # Desplegable
    try:
        # Generación de informe de perfil
        #profile = ProfileReport(df_filtered, explorative=True)
        """
        # Mostrar informe HTML dentro de Streamlit
        components.html(
            profile.to_html(),
            height=1200,
            scrolling=True
        )
        """
    
    except Exception as e:
        st.error(f"Error al generar el perfilado: {e}") # Mostrar mensaje de error

# %%
# ================================================================================
# 1.4 - DEFINICIÓN Y CÁLCULO DE KPIS
# ================================================================================

try:
    actual_month = months[1]
    actual_year = years

    # Cálculo del mes y año anterior
    if actual_month == 1:
        prev_month = 12
        prev_year = actual_year - 1
    else:
        prev_month = actual_month - 1
        prev_year = actual_year
    
    # Filtrar por el último mes y el mes anterior
    df_actual_month = df[(df['Month'] == actual_month) & (df['Year'] == actual_year)]
    df_prev_month = df[(df['Month'] == prev_month) & (df['Year'] == prev_year)]
    
    # Cálculos de KPI's
    
    # Ventas totales
    total_sales = df_actual_month['Total Sales'].sum()
    total_sales_prev = df_actual_month['Total Sales'].sum()
    # Delta de ventas
    delta_sales = (((total_sales - total_sales_prev) / total_sales_prev) * 100).round(1)

    # Unidades vendidas
    units_sold = df_actual_month['Units Sold'].sum()
    units_sold_prev = df_prev_month['Units Sold'].sum()
    # Delta de unidades vendidas
    delta_units = (((units_sold - units_sold_prev) / units_sold_prev) * 100).round(1)
    
    # Beneficio total
    total_profit = df_actual_month['Operating Profit'].sum()
    total_profit_prev = df_prev_month['Operating Profit'].sum()
    # Delta de beneficio
    delta_profit = (((total_profit - total_profit_prev) / total_profit_prev) * 100).round(1)
    
    # Margen promedio
    avg_margin = df_actual_month['Operating Margin'].mean() * 100
    avg_margin_prev = df_prev_month['Operating Margin'].mean() * 100
    # Delta de margen promedio
    delta_margin = (((avg_margin - avg_margin_prev) / avg_margin_prev) * 100).round(1)
    
    # Beneficio por unidad
    profit_per_unit = total_profit / units_sold
    profit_per_unit_prev = total_profit / units_sold_prev
    # Delta de beneficio por unidad
    delta_profit_per_unit = (((profit_per_unit - profit_per_unit_prev) / profit_per_unit_prev) * 100).round(1)
    
    # Coste por unidad
    cost_per_unit = (df_actual_month['Total Sales'].sum() - df_actual_month['Operating Profit'].sum()) / df_actual_month['Units Sold'].sum()
    cost_per_unit_prev = (df_prev_month['Total Sales'].sum() - df_prev_month['Operating Profit'].sum()) / df_prev_month['Units Sold'].sum()
    # Delta de coste por unidad
    delta_cost_per_unit = (((cost_per_unit - cost_per_unit_prev) / cost_per_unit_prev) * 100).round(1)

except Exception as e:
    st.error(f"Error al calcular las métricas: {e}") # Mostrar mensaje de error

st.subheader("Indicadores de negocio") # Subtítulo
st.caption(f"Datos mensuales, correspondientes a {actual_month}/{last_year}.") # Texto antes de las métricas

# Función para formatear millones y miles
def format_number(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.0f}K"
    else:
        return str(n)

try:
    # Mostrar métricas en columnas
    col1, col2, col3 = st.columns(3)
    
    col1.metric("📊 Ventas Totales", f"${format_number(total_sales)}", f"{delta_sales}%")
    col2.metric("📊 Unidades Vendidas", format_number(units_sold), f"{delta_units}%")

    col4, col5, col6 = st.columns(3)
    
    col4.metric("💰 Beneficio Total", f"${format_number(total_profit)}", f"{delta_profit}%")
    col5.metric("💰 Margen Promedio", f"{avg_margin:.0f}%", f"{delta_margin}%")
    col6.metric("⚙️ Coste por Unidad", f"${cost_per_unit:.0f}", f"{delta_cost_per_unit}%")

except Exception as e:
    st.error(f"Error al generar las métricas: {e}") # Mostrar mensaje de error

# %%
# ================================================================================
# 1.5 - VISUALIZACIÓN DE DATOS
# ================================================================================

st.subheader("Visualización de datos") # Subtítulo

try:
    # Ventas semanales
    weekly_sales = df.groupby(["Year", "Week Number"]).agg({"Total Sales":"sum"}).reset_index()
    # Gráfico de líneas con ventas por semana
    fig_line = px.line(weekly_sales, x="Week Number", y="Total Sales")
    fig_line.update_layout(xaxis_title="Número de semana", yaxis_title="Ventas totales") # Etiquetas a los ejes x e y
    
    # Mostrar gráfico con título
    st.write("##### Evolución histórica semanal de ventas")
    st.plotly_chart(fig_line, width='stretch')

except Exception as e:
    st.error(f"Error al visualizar: {e}") # Mostrar mensaje de error

try:
    # Beneficio total por retailer
    sales_by_method = df_filtered.groupby("Retailer").agg({"Operating Profit":"sum"}).reset_index()
    # Gráfico de tarta con el beneficio por retailer
    fig_pie = px.pie(sales_by_method, names="Retailer", values="Operating Profit")

    # Mostrar gráfico con título
    st.write("#### Distribución de beneficio por retailer")
    st.plotly_chart(fig_pie, width='stretch')

except Exception as e:
    st.error(f"Error al visualizar: {e}") # Mostrar mensaje de error

try:
    # Unidades vendidas por producto y método de venta
    units_per_product = df_filtered.groupby(["Product", "Sales Method"]).agg({"Units Sold":"sum"}).reset_index()
    # Gráfico de barras con las unidades vendidas por product y método de venta
    fig_barg = px.bar(units_per_product, x="Product", y="Units Sold", color="Sales Method")
    fig_barg.update_layout(xaxis_title="Producto", yaxis_title="Unidades vendidas", legend_title_text="Método de venta") # Etiquetas a los ejes y leyenda
    fig_barg.update_xaxes(tickangle=-30) # Girar etiquetas del eje x
    
    # Mostrar gráfico con título
    st.write("##### Volumen de unidades vendidas por producto y método de venta")
    st.plotly_chart(fig_barg, width='stretch')
    
except Exception as e:
    st.error(f"Error al visualizar: {e}") # Mostrar mensaje de error

# %%
# ================================================================================
# 2 - SISTEMA DINÁMICO Y CONVERSACIONAL
# ================================================================================

st.header("🗨️ N2. Sistema Dinámico y Conservacional", divider='gray') # Encabezado de sección

# Variables para LLM

# API Key de Gemini
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] # Extraer la clave en https://aistudio.google.com/app/api-keys y pegarla aquí
# Selección de modelo Gemini
GEMINI_MODEL = "gemini-2.5-flash"

# Variables para construir prompts

# Datos
data = df.to_string()
# Muestra de datos
data_sample = df.sample(5).to_string()
# Estructura de DataFrame
df_schema = df.dtypes.to_string()

# %%
# ================================================================================
# SECCIÓN 2.1 - EXPORTACIÓN DE DATOS FILTRADOS
# ================================================================================

with st.sidebar:
    
    st.divider() # Línea separadora
    st.subheader("Exportar datos filtrados") # Subtítulo

    # Botón de descarga
    st.download_button(label="📥 Descargar CSV",
                       data=df_filtered.to_csv(index=False), # Exportar DataFrame filtrado
                       file_name=f"data_exported.csv", # Nombre de archivo
                       mime='text/csv' # Para exportar archivos de texto plano
                      )

# %%
# ================================================================================
# 2.2 - CONEXIÓN A LLM VÍA API
# ================================================================================

st.subheader("Chat analítico ") # Subtítulo

# Inicialización de Gemini con la key directamente
try:
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Error con la clave de Gemini: {e}")

# Pregunta de usuario
question_llm = st.text_input("Escribe una pregunta sobre los datos:")

# Responder pregunta sobre los datos
try:
    if question_llm:
        # Prompt
        # Añadir enfoque con ejemplos few-shot
        prompt_llm = (f"Actúa como un experto en análisis de datos y en el sector de Adidas.\n\n"
                      f"Datos disponibles:\n{data}\n\n"
                      f"Pregunta:\n{question_llm}\n\n"
                      
                      "Intrucciones obligatorias:\n"
                      "1. Basa tu respuesta únicamente en los datos proporcionados.\n"
                      "2. Sé conciso y claro; máximo una línea de explicación intermedia si es necesaria.\n"
                      "3. Devuelve solo la información solicitada. Si es cálculo o resumen, muestra solo el resultado principal.\n"
                      "4. Evita suposiciones o datos inventados."
                      )

        with st.spinner("Gemini está procesando tu pregunta..."):
            # Procesamiento del LLM
            response_llm = client.models.generate_content(model=GEMINI_MODEL,
                                                          #temperature=0.1,
                                                          contents=prompt_llm)
            # Respuesta del LLM
            st.success("Análisis completado")
            st.write(response_llm.text)
    
except Exception as e:
    st.error(f"Error al procesar la pregunta: {e}") # Mostrar mensaje de error

# Extraer la clave de https://console.groq.com/keys y pegarla aquí:

#try:
#        with st.spinner("Llama 3.1 a través de Groq está procesando tu pregunta..."):
#            # Petición a Groq
#            completion = client.chat.completions.create(
#                model="llama-3.1-8b-instant",
#                messages=[{"role": "user", "content": prompt_analytical}],
#                temperature=0.1
#            )
#
#            st.write(completion.choices[0].message.content)
                
#except Exception as e:
#    st.error(f"Error general: {e}")

# %%
# ================================================================================
# 2.3 - EJECUCIÓN DE CONSULTAS CON SQL
# ================================================================================

st.subheader("Editor manual de consultas SQL") # Subtítulo

# Pregunta de usuario
querie = st.text_input("Escribe una query:",
                       help = "Ejemplo: SELECT * FROM ventas WHERE \"Units Sold\" > 1000")

# Botón para ejecutar querie
if st.button("Ejecutar query"):
    try:
        # Ejecución de querie
        result_sql = ps.sqldf(querie, {"ventas": df})
        
        st.success("Query ejecutada correctamente")
        st.dataframe(result_sql) # Mostrar resultados
        
    except Exception as e:
        st.error(f"Error al ejecutar la query: {e}") # Mostrar mensaje de error

# %%
# ================================================================================
# 2.4 - CONVERSIÓN DE CONSULTAS EN LENGUAJE NATURAL A SQL
# ================================================================================

st.subheader("Generador de consultas SQL") # Subtítulo

# Pregunta de usuario
question_sql = st.text_input("Escribe una consulta en lenguaje natural para convertirla en SQL:")

# Convertir pregunta en consulta SQL
try:
    if question_sql:
        # Prompt para generar querie
        prompt_sql = (f"Convierte la siguiente descripción en una consulta SQL.\n\n" 
                      f"Descripción:\n{question_sql}\n\n"
                      f"Estructura de tabla (columnas y tipos de datos):\n{df_schema}\n\n"
                      
                      "Intrucciones obligatorias:\n"
                      "1. La tabla se llama exactamente \"ventas\".\n"
                      "2. Todas las variables (columnas) deben ir entre comillas dobles \"\".\n"
                      "3. Usa únicamente las columnas proporcionadas.\n"
                      "4. Devuelve sólo el código SQL, sin texto adicional."
                     )

        with st.spinner("Gemini está procesando tu consulta..."):
            # Procesamiento del LLM
            response_sql = client.models.generate_content(model=GEMINI_MODEL, contents=prompt_sql)
            
            # Respuesta del LLM
            st.success("Consulta generada")
            st.write(response_sql.text)

except Exception as e:
    st.error(f"Error al generar la query: {e}") # Mostrar mensaje de error

# %%
# ================================================================================
# 2.5 - CONVERSIÓN DE CONSULTAS EN LENGUAJE NATURAL A SQL + EJECUCIÓN
# ================================================================================

st.subheader("Motor de consultas SQL con lenguaje natural")  # Subtítulo

# Pregunta de usuario
question_auto_sql = st.text_input("Escribe una pregunta en el motor SQL automático:") # El texto debe ser diferente a los anteriores

# Convertir pregunta en consulta SQL y ejecutarla
try:
    if question_auto_sql:
        # Prompt para generar y ejecutar querie
        prompt_auto_sql = (f"Convierte la siguiente descripción en una consulta SQL.\n\n"
                           f"Descripción:\n{question_auto_sql}\n\n"
                           f"Estructura de tabla (columnas y tipos de datos):\n{df_schema}\n\n"
                           
                           "Intrucciones obligatorias:\n"
                           "1. La tabla se llama exactamente \"ventas\".\n"
                           "2. Todas las variables (columnas) deben ir entre comillas dobles \"\".\n"
                           "3. Usa únicamente las columnas proporcionadas.\n"
                           "4. Devuelve sólo el código SQL, sin texto adicional."
                          )

        with st.spinner("Gemini está procesando tu consulta..."):
            # Procesamiento del LLM
            response_auto_sql = client.models.generate_content(model=GEMINI_MODEL, contents=prompt_auto_sql)

            # Conversión de consulta en NL a SQL
            sql_generated = response_auto_sql.text # Obtención del texto de la respuesta
            sql_generated = sql_generated.strip() # Eliminación de espacios en blanco al principio y al final
            sql_generated = sql_generated.replace("```sql", "").replace("```", "").strip() # Eliminación de bloques de código Markdown
        
            try:    
                # Ejecución de querie
                result_auto_sql = ps.sqldf(sql_generated, {"ventas": df})
    
                st.success("Consulta ejecutada correctamente")
                st.dataframe(result_auto_sql) # Mostrar resultados
    
                st.info("Consulta ejecutada:")
                st.code(sql_generated, language="sql") # Mostrar consulta

            except Exception as e_exec:
                st.error(f"Error al ejecutar la query generada: {e_exec}") # Mostrar mensaje de ejecución

except Exception as e_gen:
    st.error(f"Error al generar la query: {e_gen}") # Mostrar mensaje de error de generación

# %%
# ================================================================================
# 3 - SISTEMA MODELADO CON CONTEXTO DE NEGOCIO Y CONVERSACIONAL
# ================================================================================

st.header("🧠 N3. Sistema avanzado de analítica de negocio", divider='gray') # Encabezado de sección

# Variables para construir prompts

# Estructura de DataFrame
df_schema = df.dtypes.to_string()
# Valores únicos de variables categóricas
str_domains = df.select_dtypes(include='object').apply(lambda col: col.unique()).to_string()
# Estadísticas básicas de variables numéricas
num_ranges = df.select_dtypes(include='number').agg(['min', 'max']).to_string()
# Rangos de fechas
#date_ranges = df.select_dtypes(include='datetime').agg(['min', 'max']).to_string()

# %%
# ================================================================================
# 3.1 - EJECUCIÓN DE CÓDIGO DE PYTHON
# ================================================================================

st.subheader("Editor manual de código con Pandas") # Subtítulo

# Código
code = st.text_area("Escribe código en Pandas para ejecutar sobre el DataFrame:",
                    help="El resultado final debe asignarse a la variable 'result'.",
                    height=125)

# Botón para ejecutar el código
if st.button("Ejecutar código"):
    try:
        # Variables locales
        local_vars = {"df": df, "pd": pd, "np": np, "plt": plt, "sns": sns}

        # Ejecución de todo el código
        exec(code, {}, local_vars)
        # {} es el diccionario para que se no tenga acceso a las variables del script
        # local_vars es el diccionario donde se guardan todas las variables creadas o modificadas durante la ejecución

        # Mostrar el resultado (almacenado en una variable llamada 'result')
        result = local_vars.get("result", None)

        # Mostrar que no se ha encontrado 'result'        
        if result is None:
            st.warning("No se encontró la variable 'result'. Asegúrate de definirla.")

        # Mostrar resultados númericos        
        elif isinstance(result, (int, float, np.integer, np.floating)):
            st.success("Código ejecutado correctamente")
            st.write(result)

        # Mostrar DataFrame o Series
        elif isinstance(result, (pd.DataFrame, pd.Series)):
            st.success("Código ejecutado correctamente")
            st.dataframe(result)

        # Mostrar gráficos
        elif hasattr(result, "plot") or isinstance(result, plt.Axes):
            st.success("Gráfico generado correctamente")
            st.pyplot(result.figure if hasattr(result, "figure") else plt.gcf())
        
        else:
            st.success("Código ejecutado correctamente")
            st.write(result)

    except Exception as e:
        st.error(f"Error al ejecutar el código: {e}")

# %%
# ================================================================================
# 3.2 - CONVERSIÓN DE CONSULTAS EN LENGUAJE NATURAL A CÓDIGO
# ================================================================================

st.subheader("Generador de código básico") # Subtítulo

# Pregunta de usuario
question_code_1 = st.text_input("Escribe una consulta en lenguaje natural para convertirla en código:", key='code_1')
# Convertir pregunta en código
try:
    if question_code_1:
        # Prompt para generar código
        prompt_code_1 = (f"Convierte la siguiente descripción en código Python usando Pandas.\n\n"
                         f"Descripción:\n{question_code_1}\n\n"
                         f"Estructura de tabla (columnas y tipos de datos):\n{df_schema}\n\n"
                                             
                         "Intrucciones obligatorias:\n"
                         "1. El DataFrame se llama exactamente \"df\" y ya está cargado en memoria.\n"
                         "2. Usa únicamente Pandas para manipulación. No incluyas visualizaciones ni gráficos.\n"
                         "3. Aplica filtrado de filas, agregaciones con agg() y groupby() y ordenamiento con sort_values().\n"
                         "4. El resultado definitivo debe asignarse a la variable \"result\".\n"
                         "5. Devuelve sólo el código Python, sin texto adicional."
                        )

        with st.spinner("Gemini está procesando tu consulta..."):
            # Procesamiento del LLM
            response_code_1 = client.models.generate_content(model=GEMINI_MODEL, contents=prompt_code_1)
            
            # Respuesta del LLM
            st.success("Código generado")
            st.write(response_code_1.text)

except Exception as e:
    st.error(f"Error al generar el código: {e}") # Mostrar mensaje de error

# %%
# ================================================================================
# 3.3 - CONVERSIÓN DE CONSULTAS EN LENGUAJE NATURAL A CÓDIGO (CON VISUALIZACIÓN)
# ================================================================================

st.subheader("Generador de código avanzado") # Subtítulo

# Pregunta de usuario
question_code_2 = st.text_input("Escribe una consulta en lenguaje natural para convertirla en código:", key='code_2')

# Convertir pregunta en código
try:
    if question_code_2:
        # Prompt para generar código
        prompt_code_2 = (f"Convierte la siguiente descripción en código Python usando Pandas.\n\n"
                         f"Descripción:\n{question_code_2}\n\n"
                         f"Estructura de tabla (columnas y tipos de datos):\n{df_schema}\n\n"
                         f"Valores permitidos por columna categórica:\n{str_domains}\n\n"
                         f"Estadísticas básicas de columnas numéricas:\n{num_ranges}\n\n"
                         f"Rango de fechas:\n\n\n"
                           
                         "Intrucciones obligatorias:\n"
                         "1. El DataFrame se llama exactamente \"df\" y ya está cargado en memoria.\n"
                         "2. Usa únicamente Pandas para manipulación y Seaborn para visualización.\n"
                         "3. Aplica filtrado de filas, agregaciones con agg() y groupby() y ordenamiento con sort_values().\n"
                         "4. El resultado definitivo (DataFrame, Series, valor numérico o Axes) debe asignarse a la variable \"result\".\n"
                         "5. Devuelve sólo el código Python, sin texto adicional."
                        )

        with st.spinner("Gemini está procesando tu consulta..."):
            # Procesamiento del LLM
            response_code_2 = client.models.generate_content(model=GEMINI_MODEL, contents=prompt_code_2)
            
            # Respuesta del LLM
            st.success("Código generado")
            st.write(response_code_2.text)

except Exception as e:
    st.error(f"Error al generar el código: {e}") # Mostrar mensaje de error

# %%
# ================================================================================
# 3.4 - CARGA DE GLOSARIO DE NEGOCIO
# ================================================================================

st.subheader("Generador de código con conocimiento de negocio") # Subtítulo

# Pregunta de usuario
question_code_3 = st.text_input("Escribe una consulta en lenguaje natural para convertirla en código:", key='code_3')

# Convertir pregunta en código
try:
    if question_code_3:
        # Prompt para generar código
        prompt_code_3 = (f"Convierte la siguiente descripción en código Python usando Pandas.\n\n"
                         f"Descripción:\n{question_code_3}\n\n"
                         f"Estructura de tabla (columnas y tipos de datos):\n{df_schema}\n\n"
                         f"Valores permitidos por columna categórica:\n{str_domains}\n\n"
                         f"Estadísticas básicas de columnas numéricas:\n{num_ranges}\n\n"
                         f"Rango de fechas:\n\n\n"
                         f"Glosario de negocio:\n\n\n"
                           
                         "Intrucciones obligatorias:\n"
                         "1. El DataFrame se llama exactamente \"df\" y ya está cargado en memoria.\n"
                         "2. Usa únicamente Pandas para manipulación y Seaborn para visualización.\n"
                         "3. Aplica filtrado de filas, agregaciones con agg() y groupby() y ordenamiento con sort_values().\n"
                         "4. El resultado definitivo (DataFrame, Series, valor numérico o Axes) debe asignarse a la variable \"result\".\n"
                         "5. Devuelve sólo el código Python, sin texto adicional."
                        )

        with st.spinner("Gemini está procesando tu consulta..."):
            # Procesamiento del LLM
            response_code_3 = client.models.generate_content(model=GEMINI_MODEL, contents=prompt_code_3)
            
            # Respuesta del LLM
            st.success("Código generado")
            st.write(response_code_3.text)

except Exception as e:
    st.error(f"Error al generar el código: {e}") # Mostrar mensaje de error

# %%
# ================================================================================
# 4 - SISTEMA DE DECISIÓN ACCIONABLE
# ================================================================================

st.header("⚡ N4. Sistema de decisión accionable", divider='gray') # Encabezado de sección

# %%
# ================================================================================
# 4.1 - CONVERSIÓN DE CONSULTAS EN LENGUAJE NATURAL A CÓDIGO + EJECUCIÓN
# ================================================================================

st.subheader("Motor de código con lenguaje natural") # Subtítulo

# Pregunta de usuario
question_auto_code = st.text_input("Escribe una pregunta en el motor de código automático:")

# Convertir pregunta en código y ejecutarlo
try:
    if question_auto_code:
        # Prompt para generar y ejecutar código
        prompt_auto_code = (f"Convierte la siguiente descripción en código Python usando Pandas y Seaborn.\n\n"
                            f"Descripción:\n{question_auto_code}\n\n"
                            f"Estructura de tabla (columnas y tipos de datos):\n{df_schema}\n\n"
                            f"Valores permitidos por columna categórica:\n{str_domains}\n\n"
                            f"Estadísticas básicas de columnas numéricas:\n{num_ranges}\n\n"
                            
                            "Instrucciones obligatorias:\n"
                            "1. El DataFrame se llama exactamente \"df\" y ya está cargado en memoria.\n"
                            "2. Usa únicamente Pandas para manipulación y Seaborn para visualización.\n"
                            "3. Aplica filtrado de filas, agregaciones con agg() y groupby() y ordenamiento con sort_values().\n"
                            "4. El DataFrame, Series o valor numérico resultante debe asignarse a la variable \"result\".\n"
                            "5. Si la consulta implica comparación, tendencia o evolución, genera una gráfico. En caso contrario, no visualices.\n"
                            "6. Si se genera una visualización, el objeto Axes debe asignarse a la variable \"fig\".\n"
                            "7. Devuelve sólo el código Python, sin incluir comentarios ni explicaciones adicionales."
                           )
    
        with st.spinner("Gemini está procesando tu consulta..."):
            # Procesamiento del LLM
            response_auto_code = client.models.generate_content(model=GEMINI_MODEL, contents=prompt_auto_code)
            
            # Conversión de consulta en NL a código
            code_generated = response_auto_code.text # Obtención del texto de la respuesta
            code_generated = code_generated.strip() # Eliminación de espacios en blanco al principio y al final
            code_generated = code_generated.replace("```python", "").replace("```", "").strip() # Eliminación de bloques de código Markdown
    
        # Diccionario de variables locales para ejecutar el código
        local_vars = {"df": df, "pd": pd, "np": np, "plt": plt, "sns": sns}
    
        try:
            # Ejecución de código generado
            exec(code_generated, {}, local_vars)
            
            # Mostrar el resultado (almacenado en una variable llamada 'result')        
            result_auto = local_vars.get("result", None)
            fig_auto = local_vars.get("fig", None)
            # Guardar en session state
            st.session_state["result_auto"] = result_auto
            st.session_state["fig_auto"] = fig_auto

            # Mostrar el resultado según su tipo
            if fig_auto is not None:
                # Hay visualización
                st.success("Gráfico generado correctamente")
                st.pyplot(fig_auto.figure)
            
            else:
                # Hay DataFrame, Series o valor numérico
                if result_auto is None:
                    st.warning("No se encontró la variable 'result'")
            
                elif isinstance(result_auto, (int, float, np.integer, np.floating)):
                    st.success("Código ejecutado correctamente")
                    st.write(result_auto)
            
                elif isinstance(result_auto, (pd.DataFrame, pd.Series)):
                    st.success("Código ejecutado correctamente")
                    st.dataframe(result_auto)
            
                else:
                    st.success("Código ejecutado correctamente")
                    st.write(result_auto)

            # Mostrar código
            st.info("Código ejecutado:")
            st.code(code_generated, language="python")
            
        except Exception as e_exec:
            st.error(f"Error al ejecutar el código generado: {e_exec}") # Mostrar mensaje de ejecución
    
except Exception as e_gen:
    st.error(f"Error al generar el código: {e_gen}") # Mostrar mensaje de error de generación

# %%
# ================================================================================
# 4.2 - GENERACIÓN DE INTERPRETACIONES/CONCLUSIONES DEL RESULTADO
# ================================================================================

st.subheader("Interpretador de resultados") # Subtítulo

    # Botón para ver la interpretación de la IA
with st.expander("Interpretación de IA", expanded=False):
    try:
        # Recuperar DataFrame, Series o valor numérico generado
        result_auto = st.session_state.get("result_auto", None)
        
        if result_auto is not None:
            
            with st.spinner("La IA está interpretando el resultado..."):
                
                # Prompt para interpretar resultados
                prompt_interpret = (f"Actúa como un experto en análisis de datos y en el sector de Adidas.\n\n"
                                    f"Interpreta el siguiente resultado y explica los hallazgos en lenguaje claro, orientado a negocio y toma de decisiones.\n\n"
                                    f"Resultado:{result_auto}\n\n"
                                    
                                    "Intrucciones obligatorias:\n"
                                    "1. No describas el código.\n"
                                    "2. No repitas la tabla.\n"
                                    "3. Extrae conclusiones claras.")
        
                # Llamada al LLM
                response_interpret = client.models.generate_content(model=GEMINI_MODEL, contents=prompt_interpret)
        
                # Resultado del LLM
                st.markdown(response_interpret.text)
                # Resultado
                #st.write(result_auto)
                
    except Exception as e:
        st.error(f"Error al interpretar el resultado: {e}") # Mostrar mensaje de error

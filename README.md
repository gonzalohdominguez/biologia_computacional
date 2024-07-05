# Trabajo Final: Gonzalo Hernán Domínguez
### Manejo de datos en biología computacional. Herramientas de estadística

1. <b>Objetivo</b>: El trabajo final tiene como objetivo sistematizar y aplicar el diseño de experimentos, el manejo automatizado de datos y el análisis estadístico utilizando lenguajes de programación.
2. <b>Forma de entrega</b>: El trabajo práctico se realizará de forma individual, y se entregará el día 21 de junio. El mismo deberá además estar en un repositorio pública en Github (creado para la materia), el cual deberá contener un archivo README.md con los datos de los integrantes del equipo. Los trabajos serán presentados por los miembros del equipo en una exposición oral en los días 27 de junio o 4 de julio.
3. <b>Dataset</b>: para poder realizar el trabajo utilizarán una tabla propia la cual debe contener información tanto discreta como continua. Sobre la misma plantearemos un conjunto de hipótesis que nos lleven a comparar grupos de datos continuos y discretos.
4. <b>Pautas a seguir</b>:
    * Realizar una descripción del sistema que se intenta estudiar y de las variables medidas sobre la muestra. ¿Qué criterio se utilizó para la toma de muestra?
    * Representar de forma gráfica las distribuciones de frecuencias de las variables.
    * Registrar las medidas características de cada distribución (centralización y dispersión).
    * Estimar los intervalos de confianza.
    * Determinar el tamaño de la muestra.
    * Ensayos de hipótesis:
        * Realizar un contraste de hipótesis para dos o más poblaciones.
        * Realizar un análisis de dependencia de variables categóricas.
        * Evaluar el ajuste de una recta de regresión e interpretar el coeficiente de correlación.

### Importamos las librerías a utilizar


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import statsmodels.api as sm
from statsmodels.stats.power import TTestIndPower as smp
```

### Dataset utilizado: Qualidiab
El registro de Qualidiab es una base de datos sobre la situación real de las personas con diabetes en nuestra región. Permite determinar el grado de control clínico y metabólico de cada paciente, evaluar la frecuencia de complicaciones agudas y crónicas, analizar el acceso a la atención médica y los medicamentos, evaluar la adherencia al tratamiento y estudiar los patrones de prescripción.


```python
fichas = pd.read_excel('fichas_dm2.xlsx', sheet_name='Sheet1')
```


```python
fichas
```




<div>

    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id_ficha</th>
      <th>paciente_id</th>
      <th>cobertura_privado</th>
      <th>cobertura_obra_social</th>
      <th>cobertura_no</th>
      <th>antecedentes_hta</th>
      <th>antecedentes_dm2</th>
      <th>antecedentes_dm1</th>
      <th>antecedentes_tag</th>
      <th>antecedentes_gaa</th>
      <th>...</th>
      <th>trata_diabetes</th>
      <th>trata_dislipemia</th>
      <th>trata_insulina</th>
      <th>monoterapia_oral</th>
      <th>combinado_oral</th>
      <th>oral_con_insulina</th>
      <th>comp_macrovasculares</th>
      <th>comp_microvasculares</th>
      <th>edad_en_registro</th>
      <th>anios_con_enfermedad</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4774</td>
      <td>5599</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>77</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5876</td>
      <td>5632</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>62</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4127</td>
      <td>5710</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>53</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>411</td>
      <td>5947</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>62</td>
      <td>-3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5241</td>
      <td>5963</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>54</td>
      <td>25</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3310</th>
      <td>7708</td>
      <td>13124</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>60</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3311</th>
      <td>12847</td>
      <td>17486</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>80</td>
      <td>12</td>
    </tr>
    <tr>
      <th>3312</th>
      <td>7609</td>
      <td>13036</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>72</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3313</th>
      <td>12651</td>
      <td>17266</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>87</td>
      <td>17</td>
    </tr>
    <tr>
      <th>3314</th>
      <td>12928</td>
      <td>17563</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>57</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>3315 rows × 69 columns</p>
</div>



### Hipótesis
* Existe una correlación entre el nivel de azúcar en sangre (HbA1c) y la glucemia en ayunas.
* Se plantea una relación entre el buen control de HbA1c y la presencia de antecedentes de dislipemia.

---

## Análisis de dependencia de variables categóricas: relación entre buen control de hba1c y antecedentes de dislipemia

Hemos creado una nueva columna categórica, identificando a aquellos pacientes con un buen control metabólico como aquellos cuya HbA1c es menor o igual al 7%.


```python
def buen_control_hba1c(valor):
    if valor <= 7:
        return 1
    else:
        return 0

fichas['buen_control_hba1c'] = fichas['exploraciones_hba1c_valor'].apply(buen_control_hba1c)
```


```python
a = fichas['buen_control_hba1c'] == 1
b = fichas['antecedentes_dislipemia'] == 1
```


```python
# Comparamos los datos en una tabla de contingencia
groups = fichas.groupby([a,b]).count() 
groups['buen_control_hba1c']
```




    buen_control_hba1c  antecedentes_dislipemia
    False               False                      983
                        True                       988
    True                False                      674
                        True                       670
    Name: buen_control_hba1c, dtype: int64




```python
# Test de comparación de datos cualitativos
print(ss.chisquare(groups['buen_control_hba1c'], ddof=0, axis=0))
```

    Power_divergenceResult(statistic=118.61568627450981, pvalue=1.5330568489577969e-25)
    

Este p-valor es extremadamente pequeño , lo que indica que la probabilidad de observar una diferencia tan grande en las frecuencias observadas y esperadas por azar es extremadamente baja.

En el contexto de la prueba de Chi-cuadrado: 

* Hipótesis nula (H0): No hay asociación entre las variables buen_control_hba1c y antecedentes_dislipemia (son independientes).
* Hipótesis alternativa (H1): Hay una asociación entre las variables buen_control_hba1c y antecedentes_dislipemia (no son independientes).

Dado que el p-valor es mucho menor que cualquier nivel de significancia comúnmente usado (como 0.05, 0.01, o incluso 0.001), podemos rechazar la hipótesis nula. Esto sugiere que hay una asociación significativa entre buen_control_hba1c y antecedentes_dislipemia.

--- 

## Correlación entre hb1ac y glucemia en ayunas

##### Medidas de centralización y dispersión


```python
# Calculamos las medidas de centralización y dispersión para 'exploraciones_hba1c_valor' y para 'exploraciones_glucemia_ayunas_valor'
fichas.describe()[['exploraciones_hba1c_valor', 'exploraciones_glucemia_ayunas_valor']]
```




<div>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>exploraciones_hba1c_valor</th>
      <th>exploraciones_glucemia_ayunas_valor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3315.000000</td>
      <td>3315.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>7.811566</td>
      <td>149.556259</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>12.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>6.500000</td>
      <td>109.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7.400000</td>
      <td>133.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8.610000</td>
      <td>172.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>18.000000</td>
      <td>583.000000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.931524</td>
      <td>61.874765</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Calculamos el rango e IQR para 'exploraciones_hba1c_valor'
hba1c_range = fichas['exploraciones_hba1c_valor'].max() - fichas['exploraciones_hba1c_valor'].min()
hba1c_iqr = fichas['exploraciones_hba1c_valor'].quantile(0.75) - fichas['exploraciones_hba1c_valor'].quantile(0.25)

# Calculamos el rango e IQR para 'exploraciones_glucemia_ayunas_valor'
glucemia_range = fichas['exploraciones_glucemia_ayunas_valor'].max() - fichas['exploraciones_glucemia_ayunas_valor'].min()
glucemia_iqr = fichas['exploraciones_glucemia_ayunas_valor'].quantile(0.75) - fichas['exploraciones_glucemia_ayunas_valor'].quantile(0.25)

# Imprimir los resultados
print("Medidas de 'exploraciones_hba1c_valor':")
print(f"Rango: {hba1c_range}")
print(f"Rango intercuartílico (IQR): {hba1c_iqr}")

print("\nMedidas de 'exploraciones_glucemia_ayunas_valor':")
print(f"Rango: {glucemia_range}")
print(f"Rango intercuartílico (IQR): {glucemia_iqr}")
```

    Medidas de 'exploraciones_hba1c_valor':
    Rango: 18.0
    Rango intercuartílico (IQR): 2.1099999999999994
    
    Medidas de 'exploraciones_glucemia_ayunas_valor':
    Rango: 571
    Rango intercuartílico (IQR): 63.0
    

##### Tamaño de la muestra


```python
# Calcular la media y la desviación estándar de los datos existentes
datos_mean1 = np.mean(fichas['exploraciones_hba1c_valor'])
datos_mean2 = np.mean(fichas['exploraciones_glucemia_ayunas_valor'])
datos_std = np.std(fichas['exploraciones_glucemia_ayunas_valor'], ddof=1)

# Definir los parámetros de la prueba
effect_size = abs(datos_mean1 - datos_mean2)/ datos_std
alpha = 0.05
power = 0.8

# Crear una instancia de la clase TTestIndPower
ttp = TTestIndPower()

# Calcular el tamaño muestral necesario
n = ttp.solve_power(effect_size=effect_size, alpha=alpha, power=power, ratio=1.0, alternative='two-sided')

# Imprimir el resultado
print("El tamaño muestral necesario es:", round(n))
```

    El tamaño muestral necesario es: 4
    

##### Intervalos de confianza


```python
from scipy.stats import norm

# Función para calcular el intervalo de confianza
def calculate_confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(n)
    margin_of_error = std_err * norm.ppf((1 + confidence) / 2)
    return mean - margin_of_error, mean + margin_of_error

# Calcular el intervalo de confianza para 'exploraciones_hba1c_valor'
hba1c_ci_lower, hba1c_ci_upper = calculate_confidence_interval(fichas['exploraciones_hba1c_valor'])

# Calcular el intervalo de confianza para 'exploraciones_glucemia_ayunas_valor'
glucemia_ci_lower, glucemia_ci_upper = calculate_confidence_interval(fichas['exploraciones_glucemia_ayunas_valor'])

# Imprimir los resultados
print("Intervalo de confianza del 95% para 'exploraciones_hba1c_valor':")
print(f"({hba1c_ci_lower}, {hba1c_ci_upper})")

print("\nIntervalo de confianza del 95% para 'exploraciones_glucemia_ayunas_valor':")
print(f"({glucemia_ci_lower}, {glucemia_ci_upper})")
```

    Intervalo de confianza del 95% para 'exploraciones_hba1c_valor':
    (7.745813985971088, 7.87731723574837)
    
    Intervalo de confianza del 95% para 'exploraciones_glucemia_ayunas_valor':
    (147.44996109110343, 151.66255776259192)
    

##### Forma gráfica las distribuciones de frecuencias de las variables


```python
# Crear una figura con cuatro subplots (2 filas, 2 columnas)
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Primer gráfico (histograma de HbA1c)
axes[0, 0].hist(fichas['exploraciones_hba1c_valor'], bins=100)
axes[0, 0].set_title('Distribución de HbA1c')
axes[0, 0].set_xlabel('HbA1c')
axes[0, 0].set_ylabel('Frecuencia')

# Segundo gráfico (histograma de Glucemia en Ayunas)
axes[0, 1].hist(fichas['exploraciones_glucemia_ayunas_valor'], bins=100)
axes[0, 1].set_title('Distribución de Glucemia en Ayunas')
axes[0, 1].set_xlabel('Glucemia en Ayunas')
axes[0, 1].set_ylabel('Frecuencia')

# Tercer gráfico (boxplot de HbA1c)
sns.boxplot(y=fichas['exploraciones_hba1c_valor'], ax=axes[1, 0])
axes[1, 0].set_title('Boxplot de HbA1c')
axes[1, 0].set_ylabel('HbA1c')

# Cuarto gráfico (boxplot de Glucemia en Ayunas)
sns.boxplot(y=fichas['exploraciones_glucemia_ayunas_valor'], ax=axes[1, 1])
axes[1, 1].set_title('Boxplot de Glucemia en Ayunas')
axes[1, 1].set_ylabel('Glucemia en Ayunas')

# Ajustar el diseño para que los gráficos no se solapen
plt.tight_layout()

# Mostrar los gráficos
plt.show()
```


    
![png](output_26_0.png)
    


##### Test de normalidad


```python
print("Test de normalidad hba1c: ", ss.normaltest(fichas['exploraciones_hba1c_valor']))
print("Test de normalidad glucemia en ayunas: ", ss.normaltest(fichas['exploraciones_glucemia_ayunas_valor']))
```

    Test de normalidad hba1c:  NormaltestResult(statistic=741.6071966079205, pvalue=9.163120266007718e-162)
    Test de normalidad glucemia en ayunas:  NormaltestResult(statistic=1388.861696704899, pvalue=2.5853211384481055e-302)
    

Como las distribuciones no son normales, deberíamos utilizar un test de correlación para datos no paramétricos. Esa correlación es la prueba de Spearman. 


```python
x = fichas['exploraciones_hba1c_valor'][:]
y = fichas['exploraciones_glucemia_ayunas_valor'][:]
ss.spearmanr(x, y)
```




    SignificanceResult(statistic=0.5719014197437254, pvalue=2.6002198374445652e-287)



* En este caso, 0.5719 sugiere una correlación moderada positiva.
* Dado que este valor p es significativamente menor que cualquier nivel de significancia común (por ejemplo, 0.05), generalmente se interpreta como evidencia muy fuerte en contra de la hipótesis nula. Esto sugiere que hay una correlación significativa entre exploraciones_hba1c_valor y exploraciones_glucemia_ayunas_valor.

Los valores de HbA1c y los de glucemia en ayunas tienden a variar juntos en una dirección monotónica positiva.


```python
sns.regplot(x=x, y=y, color="blue", line_kws=dict(color="r"))
```




    <Axes: xlabel='exploraciones_hba1c_valor', ylabel='exploraciones_glucemia_ayunas_valor'>




    
![png](output_32_1.png)
    



```python
X = x
Y = y

results = sm.OLS(Y,sm.add_constant(X)).fit()

print(results.summary())
```

                                         OLS Regression Results                                    
    ===============================================================================================
    Dep. Variable:     exploraciones_glucemia_ayunas_valor   R-squared:                       0.328
    Model:                                             OLS   Adj. R-squared:                  0.328
    Method:                                  Least Squares   F-statistic:                     1616.
    Date:                                 Fri, 05 Jul 2024   Prob (F-statistic):          3.46e-288
    Time:                                         15:02:29   Log-Likelihood:                -17719.
    No. Observations:                                 3315   AIC:                         3.544e+04
    Df Residuals:                                     3313   BIC:                         3.546e+04
    Df Model:                                            1                                         
    Covariance Type:                             nonrobust                                         
    =============================================================================================
                                    coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------------------
    const                         6.2664      3.672      1.707      0.088      -0.932      13.465
    exploraciones_hba1c_valor    18.3433      0.456     40.203      0.000      17.449      19.238
    ==============================================================================
    Omnibus:                     1172.154   Durbin-Watson:                   1.839
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             8404.765
    Skew:                           1.494   Prob(JB):                         0.00
    Kurtosis:                      10.205   Cond. No.                         34.0
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    

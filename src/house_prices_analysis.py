# -*- coding: utf-8 -*-
"""
Solución Completa - Análisis de Precios de Viviendas de Ames
Resolviendo todos los problemas del ejercicio
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PROBLEMA 1: Adquirir conjunto de datos
# =============================================================================
print("=" * 70)
print("PROBLEMA 1: CARGAR DATOS CON pd.read_csv()")
print("=" * 70)

# データを読み込む (Cargar datos)
data = pd.read_csv('train.csv')
print("✓ Dataset cargado exitosamente")
print(f"✓ Dimensiones: {data.shape} (filas, columnas)")
print(f"✓ Columnas: {len(data.columns)}")
print(f"✓ Muestras: {len(data)}")

# =============================================================================
# PROBLEMA 2: Examinar el conjunto de datos
# =============================================================================
print("\n" + "=" * 70)
print("PROBLEMA 2: EXAMINAR EL CONJUNTO DE DATOS")
print("=" * 70)

print("INFORMACIÓN SOBRE EL DATASET:")
print("- Competencia: House Prices - Advanced Regression Techniques")
print("- 1460 observaciones de viviendas en Ames, Iowa")
print("- 80 características + 1 variable objetivo (SalePrice)")
print("- Objetivo: Predecir el precio final de cada vivienda")

# =============================================================================
# PROBLEMA 3: Comprobación de los datos
# =============================================================================
print("\n" + "=" * 70)
print("PROBLEMA 3: COMPROBACIÓN DE LOS DATOS")
print("=" * 70)

# データの概要を表示する (Mostrar resumen de datos)
print("1. INFORMACIÓN GENERAL DEL DATAFRAME:")
print(data.info())

# 目的変数の位置を確認 (Verificar posición de variable objetivo)
print("\n2. VARIABLE OBJETIVO (SalePrice):")
print(data['SalePrice'].describe())

# 基本統計量を確認 (Verificar estadísticas básicas)
print("\n3. ESTADÍSTICAS DESCRIPTIVAS (primeras 10 columnas numéricas):")
numeric_cols = data.select_dtypes(include=[np.number]).columns
print(data[numeric_cols[:10]].describe())
# Verificar tipos de datos
print("\n4. TIPOS DE DATOS:")
print(f"• Columnas numéricas: {len(data.select_dtypes(include=[np.number]).columns)}")
print(f"• Columnas categóricas: {len(data.select_dtypes(include=['object']).columns)}")
print(f"• Variable objetivo: SalePrice (numérica)")

# =============================================================================
# PROBLEMA 4: Manejo de valores faltantes
# =============================================================================
print("\n" + "=" * 70)
print("PROBLEMA 4: MANEJO DE VALORES FALTANTES")
print("=" * 70)

# 欠損値の可視化 (Visualizar valores faltantes)
print("1. VISUALIZACIÓN DE VALORES FALTANTES:")
plt.figure(figsize=(15, 8))
msno.matrix(data, figsize=(15, 8))
plt.title('Mapa de Valores Faltantes - Dataset House Prices', fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('missing_values.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Mapa de valores faltantes guardado como 'missing_values.png'")

# 欠損値の割合を計算する (Calcular porcentaje de valores faltantes)
print("\n2. PORCENTAJE DE VALORES FALTANTES POR COLUMNA:")
missing_ratio = (data.isnull().sum() / len(data)) * 100
missing_ratio = missing_ratio[missing_ratio > 0].sort_values(ascending=False)

print("Columnas con valores faltantes (ordenadas por porcentaje):")
for col, ratio in missing_ratio.items():
    print(f"  {col}: {ratio:.2f}% ({data[col].isnull().sum()} valores)")

# Eliminar columnas con 5 o más valores faltantes
print("\n3. ELIMINAR COLUMNAS CON 5 O MÁS VALORES FALTANTES:")
columns_to_drop = missing_ratio[missing_ratio > 0.34].index  # Aprox 5/1460 = 0.34%
print(f"Columnas a eliminar ({len(columns_to_drop)}): {list(columns_to_drop)}")

data_cleaned = data.drop(columns=columns_to_drop)

# Eliminar filas con valores faltantes
print("\n4. ELIMINAR FILAS CON VALORES FALTANTES:")
before_rows = len(data_cleaned)
data_cleaned = data_cleaned.dropna()
after_rows = len(data_cleaned)

print(f"• Filas antes: {before_rows}")
print(f"• Filas después: {after_rows}")
print(f"• Filas eliminadas: {before_rows - after_rows}")
print(f"• Porcentaje de datos conservados: {(after_rows/before_rows)*100:.1f}%")

# =============================================================================
# PROBLEMA 5: Investigación terminológica
# =============================================================================
print("\n" + "=" * 70)
print("PROBLEMA 5: INVESTIGACIÓN TERMINOLÓGICA")
print("=" * 70)

print("DEFINICIONES ESTADÍSTICAS:")
print("• CURTOSIS (Kurtosis):")
print("  - Mide el 'pico' de una distribución comparada con la normal")
print("  - >0: Distribución más picuda (leptocúrtica)")
print("  - <0: Distribución más plana (platicúrtica)")
print("  - =0: Similar a distribución normal (mesocúrtica)")

print("\n• ASIMETRÍA (Skewness):")
print("  - Mide la simetría de una distribución")
print("  - >0: Cola hacia la derecha (asimetría positiva)")
print("  - <0: Cola hacia la izquierda (asimetría negativa)")
print("  - =0: Distribución simétrica")

# =============================================================================
# PROBLEMA 6: Comprobación de la distribución
# =============================================================================
print("\n" + "=" * 70)
print("PROBLEMA 6: COMPROBACIÓN DE LA DISTRIBUCIÓN")
print("=" * 70)

# 顕示変数の分布確認 (Verificar distribución de variable objetivo)
print("1. DISTRIBUCIÓN ORIGINAL DE SalePrice:")

# 尖度と歪度の算出 (Calcular curtosis y asimetría)
original_kurtosis = data_cleaned['SalePrice'].kurtosis()
original_skewness = data_cleaned['SalePrice'].skew()

print(f"• Curtosis: {original_kurtosis:.4f}")
print(f"• Asimetría: {original_skewness:.4f}")
print(f"• Media: ${data_cleaned['SalePrice'].mean():,.2f}")
print(f"• Mediana: ${data_cleaned['SalePrice'].median():,.2f}")

# Visualización distribución original
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(data_cleaned['SalePrice'], kde=True, bins=50)
plt.title(f'Distribución Original\nCurtosis: {original_kurtosis:.2f}, Asimetría: {original_skewness:.2f}')
plt.xlabel('Precio de Venta ($)')
plt.ylabel('Frecuencia')
plt.grid(True, alpha=0.3)

# 対数変換 (Transformación logarítmica)
print("\n2. TRANSFORMACIÓN LOGARÍTMICA:")
log_price = np.log(data_cleaned['SalePrice'])

# 尖度と歪度の算出 (Calcular curtosis y asimetría después de transformación)
log_kurtosis = log_price.kurtosis()
log_skewness = log_price.skew()

print(f"• Curtosis (log): {log_kurtosis:.4f}")
print(f"• Asimetría (log): {log_skewness:.4f}")
print(f"• Media (log): {log_price.mean():.4f}")
print(f"• Mediana (log): {log_price.median():.4f}")

# Visualización distribución logarítmica
plt.subplot(1, 2, 2)
sns.histplot(log_price, kde=True, bins=50)
plt.title(f'Distribución Logarítmica\nCurtosis: {log_kurtosis:.2f}, Asimetría: {log_skewness:.2f}')
plt.xlabel('log(Precio de Venta)')
plt.ylabel('Frecuencia')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('distribucion_precios.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Análisis de distribución guardado como 'distribucion_precios.png'")

# Comparación con distribución normal
print("\n3. COMPARACIÓN CON DISTRIBUCIÓN NORMAL:")
print("• ANTES de transformación logarítmica:")
print(f"  - La distribución tiene asimetría positiva ({original_skewness:.2f})")
print(f"  - Es más picuda que la normal (curtosis: {original_kurtosis:.2f})")

print("• DESPUÉS de transformación logarítmica:")
print(f"  - La asimetría se redujo a {log_skewness:.2f}")
print(f"  - La curtosis se redujo a {log_kurtosis:.2f}")
print("  - La distribución se acerca más a una distribución normal")

# =============================================================================
# PROBLEMA 7: Comprobación del coeficiente de correlación
# =============================================================================
print("\n" + "=" * 70)
print("PROBLEMA 7: COEFICIENTES DE CORRELACIÓN")
print("=" * 70)

# 相関行列の計算 (Calcular matriz de correlación)
print("1. MATRIZ DE CORRELACIÓN COMPLETA:")
numeric_data = data_cleaned.select_dtypes(include=[np.number])
corr_matrix = numeric_data.corr()

plt.figure(figsize=(16, 14))
sns.heatmap(corr_matrix, cmap='RdBu_r', center=0, square=True, 
            cbar_kws={"shrink": .8}, annot=False)
plt.title('Matriz de Correlación - Variables Numéricas', 
          fontweight='bold', pad=20, fontsize=16)
plt.tight_layout()
plt.savefig('matriz_correlacion_completa.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Matriz de correlación completa guardada como 'matriz_correlacion_completa.png'")

# 上位10個の相関の高い特徴量を表示 (Top 10 características más correlacionadas)
print("\n2. TOP 10 CARACTERÍSTICAS MÁS CORRELACIONADAS CON SalePrice:")
sale_price_corr = corr_matrix['SalePrice'].sort_values(ascending=False)
top_10_features = sale_price_corr[1:11]  # Excluir SalePrice itself

print("Ranking de correlación con SalePrice:")
for i, (feature, corr_value) in enumerate(top_10_features.items(), 1):
    print(f"  {i:2d}. {feature:20} : {corr_value:.4f}")

# Heatmap de top 10 características
top_10_names = top_10_features.index.tolist()
top_10_names.append('SalePrice')

plt.figure(figsize=(12, 10))
sns.heatmap(numeric_data[top_10_names].corr(), annot=True, cmap='RdBu_r', 
            center=0, square=True, fmt=".3f", cbar_kws={"shrink": .8})
plt.title('Top 10 Características más Correlacionadas con SalePrice', 
          fontweight='bold', pad=20, fontsize=16)
plt.tight_layout()
plt.savefig('top10_correlaciones.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Top 10 correlaciones guardada como 'top10_correlaciones.png'")

# Descripción de características importantes
print("\n3. DESCRIPCIÓN DE CARACTERÍSTICAS IMPORTANTES:")
descriptions = {
    'OverallQual': 'Calidad general de materiales y acabados (1=Muy Pobre, 10=Excelente)',
    'GrLivArea': 'Área habitable sobre nivel del suelo (pies cuadrados)',
    'GarageCars': 'Capacidad del garaje en número de coches',
    'GarageArea': 'Tamaño del garaje en pies cuadrados',
    'TotalBsmtSF': 'Área total del sótano en pies cuadrados',
    '1stFlrSF': 'Área del primer piso en pies cuadrados',
    'FullBath': 'Número de baños completos arriba del nivel del suelo',
    'TotRmsAbvGrd': 'Número total de habitaciones arriba del nivel del suelo',
    'YearBuilt': 'Año original de construcción',
    'YearRemodAdd': 'Año de remodelación (igual a construcción si no remodelado)'
}

print("Significado de las características más importantes:")
for i, feature in enumerate(top_10_features.index, 1):
    if feature in descriptions:
        print(f"{i:2d}. {feature:15} : {descriptions[feature]}")
    else:
        print(f"{i:2d}. {feature:15} : [Información no disponible]")

# Encontrar pares altamente correlacionados
print("\n4. TOP 3 PARES DE CARACTERÍSTICAS ALTAMENTE CORRELACIONADAS:")
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.7 and corr_matrix.columns[i] != 'SalePrice' and corr_matrix.columns[j] != 'SalePrice':
            high_corr_pairs.append((
                corr_matrix.columns[i],
                corr_matrix.columns[j],
                corr_matrix.iloc[i, j]
            ))

# Ordenar por correlación descendente
high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

print("Pares con correlación > 0.7 (excluyendo SalePrice):")
for i, (feat1, feat2, corr) in enumerate(high_corr_pairs[:3], 1):
    print(f"  {i}. {feat1:15} - {feat2:15} : {corr:.4f}")

# =============================================================================
# RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 70)
print("RESUMEN EJECUTIVO - ANÁLISIS COMPLETADO")
print("=" * 70)

print("📊 RESULTADOS PRINCIPALES:")
print(f"• Dataset procesado: {data_cleaned.shape[0]} filas, {data_cleaned.shape[1]} columnas")
print(f"• Característica más importante: {top_10_features.index[0]} (corr: {top_10_features.iloc[0]:.3f})")
print(f"• Asimetría original: {original_skewness:.2f} → Log: {log_skewness:.2f}")
print(f"• Curtosis original: {original_kurtosis:.2f} → Log: {log_kurtosis:.2f}")

print("\n🔍 INSIGHTS CLAVE:")
print("1. La calidad general (OverallQual) es el mejor predictor de precio")
print("2. El área habitable (GrLivArea) tiene alta correlación con el precio")
print("3. La transformación logarítmica normaliza la distribución")
print("4. Existen características redundantes (alta correlación entre sí)")

print("\n📁 ARCHIVOS GENERADOS:")
print("   • missing_values.png - Mapa de valores faltantes")
print("   • distribucion_precios.png - Análisis de distribución")
print("   • matriz_correlacion_completa.png - Correlaciones completas")
print("   • top10_correlaciones.png - Top 10 características")

print("\n✅ TODOS LOS PROBLEMAS RESUELTOS EXITOSAMENTE")
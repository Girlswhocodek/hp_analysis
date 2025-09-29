# -*- coding: utf-8 -*-
"""
Análisis de Precios de Viviendas - Ames, Iowa
Análisis exploratorio de datos (EDA) completo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def problema_1_cargar_datos():
    """
    Problema 1: Cargar el dataset de precios de viviendas
    """
    print("=" * 70)
    print("PROBLEMA 1: CARGA DE DATOS")
    print("=" * 70)
    
    try:
        # Intentar cargar el dataset real
        data = pd.read_csv('train.csv')
        print("✓ Dataset 'train.csv' cargado exitosamente")
    except FileNotFoundError:
        print("⚠️  Archivo 'train.csv' no encontrado. Creando dataset de ejemplo...")
        data = crear_dataset_ejemplo()
    
    print(f"✓ Dimensiones del dataset: {data.shape}")
    print(f"✓ Número de características: {data.shape[1]}")
    print(f"✓ Número de muestras: {data.shape[0]}")
    
    return data

def crear_dataset_ejemplo():
    """
    Crear dataset de ejemplo si no se encuentra el archivo real
    """
    np.random.seed(42)
    n_samples = 1460
    
    # Crear dataset sintético con características similares al real
    data = pd.DataFrame({
        'Id': range(1, n_samples + 1),
        'MSSubClass': np.random.choice([20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190], n_samples),
        'MSZoning': np.random.choice(['RL', 'RM', 'C (all)', 'FV', 'RH'], n_samples, p=[0.7, 0.2, 0.05, 0.03, 0.02]),
        'LotFrontage': np.random.normal(70, 20, n_samples).clip(20, 200),
        'LotArea': np.random.normal(10000, 5000, n_samples).clip(1000, 50000),
        'Street': np.random.choice(['Pave', 'Grvl'], n_samples, p=[0.98, 0.02]),
        'Alley': np.random.choice([np.nan, 'Grvl', 'Pave'], n_samples, p=[0.93, 0.04, 0.03]),
        'LotShape': np.random.choice(['Reg', 'IR1', 'IR2', 'IR3'], n_samples, p=[0.6, 0.3, 0.08, 0.02]),
        'LandContour': np.random.choice(['Lvl', 'Bnk', 'HLS', 'Low'], n_samples, p=[0.8, 0.1, 0.07, 0.03]),
        'Utilities': np.random.choice(['AllPub', 'NoSeWa'], n_samples, p=[0.99, 0.01]),
        'LotConfig': np.random.choice(['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3'], n_samples, p=[0.7, 0.15, 0.05, 0.07, 0.03]),
        'LandSlope': np.random.choice(['Gtl', 'Mod', 'Sev'], n_samples, p=[0.9, 0.08, 0.02]),
        'Neighborhood': np.random.choice(['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst', 'Gilbert', 'NridgHt', 'Sawyer', 'NWAmes', 'SawyerW', 'BrkSide', 'Crawfor', 'Mitchel', 'NoRidge', 'Timber', 'IDOTRR', 'ClearCr', 'StoneBr', 'SWISU', 'MeadowV', 'Blmngtn', 'BrDale', 'Veenker', 'NPkVill', 'Blueste'], n_samples),
        'Condition1': np.random.choice(['Norm', 'Feedr', 'PosN', 'Artery', 'RRAe', 'RRNn', 'RRAn', 'PosA', 'RRNe'], n_samples, p=[0.8, 0.08, 0.03, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01]),
        'Condition2': np.random.choice(['Norm', 'Feedr', 'PosN', 'Artery', 'RRAe', 'RRNn', 'RRAn', 'PosA', 'RRNe'], n_samples, p=[0.95, 0.02, 0.01, 0.01, 0.005, 0.005, 0.005, 0.005, 0.005]),
        'BldgType': np.random.choice(['1Fam', '2FmCon', 'Duplex', 'TwnhsE', 'Twnhs'], n_samples, p=[0.8, 0.1, 0.05, 0.03, 0.02]),
        'HouseStyle': np.random.choice(['1Story', '2Story', '1.5Fin', 'SLvl', 'SFoyer', '1.5Unf', '2.5Unf', '2.5Fin'], n_samples, p=[0.4, 0.3, 0.1, 0.08, 0.06, 0.03, 0.02, 0.01]),
        'OverallQual': np.random.randint(1, 11, n_samples),
        'OverallCond': np.random.randint(1, 11, n_samples),
        'YearBuilt': np.random.randint(1870, 2011, n_samples),
        'YearRemodAdd': np.random.randint(1950, 2011, n_samples),
        'RoofStyle': np.random.choice(['Gable', 'Hip', 'Gambrel', 'Mansard', 'Flat', 'Shed'], n_samples, p=[0.75, 0.2, 0.02, 0.01, 0.01, 0.01]),
        'RoofMatl': np.random.choice(['CompShg', 'WdShngl', 'Metal', 'WdShake', 'Membran', 'Tar&Grv', 'Roll', 'ClyTile'], n_samples, p=[0.98, 0.005, 0.005, 0.005, 0.002, 0.001, 0.001, 0.001]),
        'Exterior1st': np.random.choice(['VinylSd', 'MetalSd', 'Wd Sdng', 'HdBoard', 'BrkFace', 'WdShing', 'CemntBd', 'Plywood', 'AsbShng', 'Stucco', 'BrkComm', 'AsphShn', 'Stone', 'ImStucc', 'CBlock'], n_samples),
        'Exterior2nd': np.random.choice(['VinylSd', 'MetalSd', 'Wd Sdng', 'HdBoard', 'Plywood', 'Wd Shng', 'CmentBd', 'BrkFace', 'Stucco', 'AsbShng', 'Brk Cmn', 'ImStucc', 'AsphShn', 'Stone', 'Other', 'CBlock'], n_samples),
        'MasVnrType': np.random.choice([None, 'BrkFace', 'Stone', 'BrkCmn'], n_samples, p=[0.6, 0.2, 0.15, 0.05]),
        'MasVnrArea': np.random.exponential(100, n_samples).clip(0, 1000),
        'ExterQual': np.random.choice(['Ex', 'Gd', 'TA', 'Fa', 'Po'], n_samples, p=[0.05, 0.3, 0.6, 0.04, 0.01]),
        'ExterCond': np.random.choice(['Ex', 'Gd', 'TA', 'Fa', 'Po'], n_samples, p=[0.02, 0.1, 0.8, 0.07, 0.01]),
        'Foundation': np.random.choice(['PConc', 'CBlock', 'BrkTil', 'Wood', 'Slab', 'Stone'], n_samples, p=[0.5, 0.4, 0.05, 0.03, 0.015, 0.005]),
        'BsmtQual': np.random.choice([None, 'Ex', 'Gd', 'TA', 'Fa', 'Po'], n_samples, p=[0.03, 0.1, 0.3, 0.5, 0.06, 0.01]),
        'BsmtCond': np.random.choice([None, 'Ex', 'Gd', 'TA', 'Fa', 'Po'], n_samples, p=[0.03, 0.02, 0.1, 0.8, 0.04, 0.01]),
        'BsmtExposure': np.random.choice([None, 'Gd', 'Av', 'Mn', 'No'], n_samples, p=[0.03, 0.1, 0.2, 0.1, 0.57]),
        'BsmtFinType1': np.random.choice([None, 'GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf'], n_samples, p=[0.03, 0.2, 0.1, 0.1, 0.1, 0.1, 0.37]),
        'BsmtFinSF1': np.random.exponential(400, n_samples).clip(0, 2000),
        'BsmtFinType2': np.random.choice([None, 'GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf'], n_samples, p=[0.8, 0.05, 0.03, 0.03, 0.03, 0.03, 0.03]),
        'BsmtFinSF2': np.random.exponential(50, n_samples).clip(0, 1000),
        'BsmtUnfSF': np.random.exponential(500, n_samples).clip(0, 1500),
        'TotalBsmtSF': np.random.normal(1000, 400, n_samples).clip(0, 3000),
        'Heating': np.random.choice(['GasA', 'GasW', 'Grav', 'Wall', 'OthW', 'Floor'], n_samples, p=[0.95, 0.03, 0.01, 0.005, 0.003, 0.002]),
        'HeatingQC': np.random.choice(['Ex', 'Gd', 'TA', 'Fa', 'Po'], n_samples, p=[0.1, 0.3, 0.5, 0.08, 0.02]),
        'CentralAir': np.random.choice(['Y', 'N'], n_samples, p=[0.95, 0.05]),
        'Electrical': np.random.choice(['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix'], n_samples, p=[0.9, 0.05, 0.03, 0.015, 0.005]),
        '1stFlrSF': np.random.normal(1200, 400, n_samples).clip(300, 3000),
        '2ndFlrSF': np.random.normal(600, 400, n_samples).clip(0, 2000),
        'LowQualFinSF': np.random.exponential(10, n_samples).clip(0, 500),
        'GrLivArea': np.random.normal(1500, 500, n_samples).clip(500, 4000),
        'BsmtFullBath': np.random.choice([0, 1, 2, 3], n_samples, p=[0.4, 0.5, 0.09, 0.01]),
        'BsmtHalfBath': np.random.choice([0, 1, 2], n_samples, p=[0.8, 0.18, 0.02]),
        'FullBath': np.random.choice([0, 1, 2, 3], n_samples, p=[0.05, 0.4, 0.5, 0.05]),
        'HalfBath': np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.35, 0.05]),
        'BedroomAbvGr': np.random.choice([0, 1, 2, 3, 4, 5, 6], n_samples, p=[0.01, 0.05, 0.3, 0.4, 0.2, 0.03, 0.01]),
        'KitchenAbvGr': np.random.choice([0, 1, 2, 3], n_samples, p=[0.01, 0.8, 0.18, 0.01]),
        'KitchenQual': np.random.choice(['Ex', 'Gd', 'TA', 'Fa', 'Po'], n_samples, p=[0.1, 0.4, 0.45, 0.04, 0.01]),
        'TotRmsAbvGrd': np.random.randint(2, 12, n_samples),
        'Functional': np.random.choice(['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal'], n_samples, p=[0.9, 0.03, 0.02, 0.02, 0.01, 0.01, 0.005, 0.005]),
        'Fireplaces': np.random.choice([0, 1, 2, 3], n_samples, p=[0.5, 0.4, 0.09, 0.01]),
        'FireplaceQu': np.random.choice([None, 'Ex', 'Gd', 'TA', 'Fa', 'Po'], n_samples, p=[0.5, 0.05, 0.15, 0.2, 0.07, 0.03]),
        'GarageType': np.random.choice([None, 'Attchd', 'Detchd', 'BuiltIn', 'Basment', 'CarPort', '2Types'], n_samples, p=[0.06, 0.6, 0.25, 0.05, 0.02, 0.015, 0.005]),
        'GarageYrBlt': np.random.randint(1900, 2011, n_samples),
        'GarageFinish': np.random.choice([None, 'Fin', 'RFn', 'Unf'], n_samples, p=[0.06, 0.4, 0.3, 0.24]),
        'GarageCars': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.06, 0.3, 0.5, 0.13, 0.01]),
        'GarageArea': np.random.normal(500, 200, n_samples).clip(0, 1200),
        'GarageQual': np.random.choice([None, 'Ex', 'Gd', 'TA', 'Fa', 'Po'], n_samples, p=[0.06, 0.02, 0.1, 0.75, 0.06, 0.01]),
        'GarageCond': np.random.choice([None, 'Ex', 'Gd', 'TA', 'Fa', 'Po'], n_samples, p=[0.06, 0.02, 0.1, 0.75, 0.06, 0.01]),
        'PavedDrive': np.random.choice(['Y', 'P', 'N'], n_samples, p=[0.9, 0.05, 0.05]),
        'WoodDeckSF': np.random.exponential(50, n_samples).clip(0, 500),
        'OpenPorchSF': np.random.exponential(40, n_samples).clip(0, 400),
        'EnclosedPorch': np.random.exponential(20, n_samples).clip(0, 300),
        '3SsnPorch': np.random.exponential(5, n_samples).clip(0, 200),
        'ScreenPorch': np.random.exponential(15, n_samples).clip(0, 300),
        'PoolArea': np.random.exponential(5, n_samples).clip(0, 400),
        'PoolQC': np.random.choice([None, 'Ex', 'Gd', 'TA', 'Fa'], n_samples, p=[0.995, 0.002, 0.001, 0.001, 0.001]),
        'Fence': np.random.choice([None, 'MnPrv', 'GdWo', 'GdPrv', 'MnWw'], n_samples, p=[0.8, 0.08, 0.06, 0.04, 0.02]),
        'MiscFeature': np.random.choice([None, 'Shed', 'Gar2', 'Othr', 'TenC'], n_samples, p=[0.95, 0.03, 0.01, 0.005, 0.005]),
        'MiscVal': np.random.exponential(100, n_samples).clip(0, 5000),
        'MoSold': np.random.randint(1, 13, n_samples),
        'YrSold': np.random.choice([2006, 2007, 2008, 2009, 2010], n_samples),
        'SaleType': np.random.choice(['WD', 'CWD', 'VWD', 'New', 'COD', 'Con', 'ConLw', 'ConLI', 'ConLD', 'Oth'], n_samples, p=[0.85, 0.03, 0.02, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005, 0.005]),
        'SaleCondition': np.random.choice(['Normal', 'Abnorml', 'Partial', 'AdjLand', 'Alloca', 'Family'], n_samples, p=[0.85, 0.08, 0.05, 0.01, 0.005, 0.005]),
        'SalePrice': np.random.lognormal(12, 0.4, n_samples)  # Precio de venta
    })
    
    # Asegurar que SalePrice sea la última columna
    sale_price = data.pop('SalePrice')
    data['SalePrice'] = sale_price
    
    print("✓ Dataset de ejemplo creado exitosamente")
    return data

def problema_2_3_investigacion_datos(data):
    """
    Problema 2 y 3: Investigación y exploración inicial del dataset
    """
    print("\n" + "=" * 70)
    print("PROBLEMA 2 y 3: INVESTIGACIÓN DEL DATASET")
    print("=" * 70)
    
    # Información general del dataset
    print("INFORMACIÓN GENERAL:")
    print(data.info())
    
    # Descripción de variables numéricas
    print("\nESTADÍSTICAS DESCRIPTIVAS (Variables Numéricas):")
    print(data.describe())
    
    # Variables categóricas
    categorical_cols = data.select_dtypes(include=['object']).columns
    print(f"\nVARIABLES CATEGÓRICAS ({len(categorical_cols)}):")
    print(list(categorical_cols))
    
    # Variables numéricas
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    print(f"\nVARIABLES NUMÉRICAS ({len(numerical_cols)}):")
    print(list(numerical_cols))
    
    # Variable objetivo
    print(f"\nVARIABLE OBJETIVO (SalePrice):")
    print(data['SalePrice'].describe())
    
    return categorical_cols, numerical_cols

def problema_4_manejo_valores_faltantes(data):
    """
    Problema 4: Identificación y manejo de valores faltantes
    """
    print("\n" + "=" * 70)
    print("PROBLEMA 4: MANEJO DE VALORES FALTANTES")
    print("=" * 70)
    
    # Calcular porcentaje de valores faltantes por columna
    missing_ratio = (data.isnull().sum() / len(data)) * 100
    missing_ratio = missing_ratio[missing_ratio > 0].sort_values(ascending=False)
    
    print("PORCENTAJE DE VALORES FALTANTES POR COLUMNA:")
    for col, ratio in missing_ratio.items():
        print(f"  {col}: {ratio:.2f}% ({data[col].isnull().sum()} valores faltantes)")
    
    # Visualizar valores faltantes
    plt.figure(figsize=(15, 8))
    msno.matrix(data)
    plt.title('Mapa de Valores Faltantes', fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('valores_faltantes.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Mapa de valores faltantes guardado como 'valores_faltantes.png'")
    
    # Eliminar columnas con más de 5 valores faltantes
    columns_to_drop = missing_ratio[missing_ratio > 0.34].index  # ~5 valores en 1460 muestras
    print(f"\nELIMINANDO COLUMNAS CON MÁS DE 5 VALORES FALTANTES:")
    print(f"Columnas a eliminar: {list(columns_to_drop)}")
    
    data_cleaned = data.drop(columns=columns_to_drop)
    
    # Eliminar filas con valores faltantes en las columnas restantes
    before_rows = len(data_cleaned)
    data_cleaned = data_cleaned.dropna()
    after_rows = len(data_cleaned)
    
    print(f"\nELIMINACIÓN DE FILAS CON VALORES FALTANTES:")
    print(f"Filas antes: {before_rows}")
    print(f"Filas después: {after_rows}")
    print(f"Filas eliminadas: {before_rows - after_rows}")
    
    return data_cleaned

def problema_5_6_distribucion_variable_objetivo(data):
    """
    Problema 5 y 6: Análisis de la distribución de la variable objetivo
    """
    print("\n" + "=" * 70)
    print("PROBLEMA 5 y 6: DISTRIBUCIÓN DE LA VARIABLE OBJETIVO")
    print("=" * 70)
    
    sale_price = data['SalePrice']
    
    # Cálculo de curtosis y asimetría original
    kurtosis_original = sale_price.kurtosis()
    skewness_original = sale_price.skew()
    
    print("ESTADÍSTICAS DE DISTRIBUCIÓN ORIGINAL:")
    print(f"• Curtosis: {kurtosis_original:.4f}")
    print(f"• Asimetría: {skewness_original:.4f}")
    print(f"• Media: ${sale_price.mean():.2f}")
    print(f"• Mediana: ${sale_price.median():.2f}")
    print(f"• Desviación estándar: ${sale_price.std():.2f}")
    
    # Transformación logarítmica
    log_sale_price = np.log(sale_price)
    kurtosis_log = log_sale_price.kurtosis()
    skewness_log = log_sale_price.skew()
    
    print("\nESTADÍSTICAS DESPUÉS DE TRANSFORMACIÓN LOGARÍTMICA:")
    print(f"• Curtosis: {kurtosis_log:.4f}")
    print(f"• Asimetría: {skewness_log:.4f}")
    print(f"• Media: {log_sale_price.mean():.4f}")
    print(f"• Mediana: {log_sale_price.median():.4f}")
    
    # Visualización comparativa
    plt.figure(figsize=(15, 10))
    
    # Distribución original
    plt.subplot(2, 2, 1)
    sns.histplot(sale_price, kde=True, bins=50)
    plt.title(f'Distribución Original\nCurtosis: {kurtosis_original:.2f}, Asimetría: {skewness_original:.2f}', fontweight='bold')
    plt.xlabel('Precio de Venta ($)')
    plt.ylabel('Frecuencia')
    plt.grid(True, alpha=0.3)
    
    # Q-Q plot original
    plt.subplot(2, 2, 2)
    stats.probplot(sale_price, dist="norm", plot=plt)
    plt.title('Q-Q Plot - Distribución Original', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Distribución logarítmica
    plt.subplot(2, 2, 3)
    sns.histplot(log_sale_price, kde=True, bins=50)
    plt.title(f'Distribución Logarítmica\nCurtosis: {kurtosis_log:.2f}, Asimetría: {skewness_log:.2f}', fontweight='bold')
    plt.xlabel('log(Precio de Venta)')
    plt.ylabel('Frecuencia')
    plt.grid(True, alpha=0.3)
    
    # Q-Q plot logarítmico
    plt.subplot(2, 2, 4)
    stats.probplot(log_sale_price, dist="norm", plot=plt)
    plt.title('Q-Q Plot - Distribución Logarítmica', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('distribucion_precios.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Análisis de distribución guardado como 'distribucion_precios.png'")
    
    # Explicación de los resultados
    print("\n📊 INTERPRETACIÓN DE RESULTADOS:")
    print("• CURTOSIS: Mide el 'pico' de la distribución")
    print("  - >0: Distribución más picuda que la normal (leptocúrtica)")
    print("  - <0: Distribución más plana que la normal (platicúrtica)")
    print("  - =0: Similar a distribución normal (mesocúrtica)")
    
    print("\n• ASIMETRÍA: Mide la simetría de la distribución")
    print("  - >0: Cola hacia la derecha (asimetría positiva)")
    print("  - <0: Cola hacia la izquierda (asimetría negativa)")
    print("  - =0: Distribución simétrica")
    
    print(f"\n💡 CONCLUSIÓN:")
    print(f"La transformación logarítmica redujo la asimetría de {skewness_original:.2f} a {skewness_log:.2f}")
    print(f"y la curtosis de {kurtosis_original:.2f} a {kurtosis_log:.2f}")
    print("Haciendo la distribución más cercana a una distribución normal")

def problema_7_analisis_correlaciones(data):
    """
    Problema 7: Análisis de correlaciones entre variables
    """
    print("\n" + "=" * 70)
    print("PROBLEMA 7: ANÁLISIS DE CORRELACIONES")
    print("=" * 70)
    
    # Calcular matriz de correlación solo para variables numéricas
    numeric_data = data.select_dtypes(include=[np.number])
    corr_matrix = numeric_data.corr()
    
    # Variables más correlacionadas con SalePrice
    sale_price_corr = corr_matrix['SalePrice'].sort_values(ascending=False)
    top_10_features = sale_price_corr[1:11]  # Excluir SalePrice itself
    
    print("TOP 10 CARACTERÍSTICAS MÁS CORRELACIONADAS CON EL PRECIO:")
    for feature, corr_value in top_10_features.items():
        print(f"  {feature}: {corr_value:.4f}")
    
    # Heatmap de correlaciones general
    plt.figure(figsize=(16, 14))
    sns.heatmap(corr_matrix, cmap='RdBu_r', center=0, square=True, 
                cbar_kws={"shrink": .8}, annot=False)
    plt.title('Matriz de Correlación - Todas las Variables Numéricas', 
              fontweight='bold', pad=20, fontsize=14)
    plt.tight_layout()
    plt.savefig('matriz_correlacion_completa.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Matriz de correlación completa guardada como 'matriz_correlacion_completa.png'")
    
    # Heatmap de las 10 características más correlacionadas
    top_10_names = top_10_features.index.tolist()
    top_10_names.append('SalePrice')  # Añadir la variable objetivo
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_data[top_10_names].corr(), annot=True, cmap='RdBu_r', 
                center=0, square=True, fmt=".2f", cbar_kws={"shrink": .8})
    plt.title('Top 10 Características más Correlacionadas con el Precio', 
              fontweight='bold', pad=20, fontsize=14)
    plt.tight_layout()
    plt.savefig('top10_correlaciones.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Top 10 correlaciones guardado como 'top10_correlaciones.png'")
    
    # Identificar pares de características altamente correlacionadas
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
    
    print("\nTOP 3 PARES DE CARACTERÍSTICAS ALTAMENTE CORRELACIONADAS:")
    for i, (feat1, feat2, corr) in enumerate(high_corr_pairs[:3], 1):
        print(f"  {i}. {feat1} - {feat2}: {corr:.4f}")
    
    return top_10_features, high_corr_pairs[:3]

def descripcion_caracteristicas_importantes(top_features):
    """
    Descripción en español de las características más importantes
    """
    print("\n" + "=" * 70)
    print("DESCRIPCIÓN DE CARACTERÍSTICAS IMPORTANTES")
    print("=" * 70)
    
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
    
    print("SIGNIFICADO DE LAS CARACTERÍSTICAS MÁS IMPORTANTES:")
    for feature in top_features.index:
        if feature in descriptions:
            print(f"• {feature}: {descriptions[feature]}")
        else:
            print(f"• {feature}: [Descripción no disponible]")

def main():
    """
    Función principal del análisis
    """
    print("🏠 ANÁLISIS DE PRECIOS DE VIVIENDAS - AMES, IOWA")
    print("=" * 70)
    
    try:
        # Problema 1: Cargar datos
        data = problema_1_cargar_datos()
        
        # Problema 2-3: Investigación inicial
        categorical_cols, numerical_cols = problema_2_3_investigacion_datos(data)
        
        # Problema 4: Manejo de valores faltantes
        data_cleaned = problema_4_manejo_valores_faltantes(data)
        
        # Problema 5-6: Análisis de distribución
        problema_5_6_distribucion_variable_objetivo(data_cleaned)
        
        # Problema 7: Análisis de correlaciones
        top_features, high_corr_pairs = problema_7_analisis_correlaciones(data_cleaned)
        
        # Descripción de características importantes
        descripcion_caracteristicas_importantes(top_features)
        
        # Resumen final
        print("\n" + "=" * 70)
        print("✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
        print("=" * 70)
        print("📊 RESUMEN EJECUTIVO:")
        print(f"• Dataset original: {data.shape[0]} filas, {data.shape[1]} columnas")
        print(f"• Dataset limpio: {data_cleaned.shape[0]} filas, {data_cleaned.shape[1]} columnas")
        print(f"• Característica más importante: {top_features.index[0]} (corr: {top_features.iloc[0]:.3f})")
        print(f"• Variables categóricas: {len(categorical_cols)}")
        print(f"• Variables numéricas: {len(numerical_cols)}")
        
        print("\n📁 ARCHIVOS GENERADOS:")
        print("   • valores_faltantes.png - Mapa de valores faltantes")
        print("   • distribucion_precios.png - Análisis de distribución")
        print("   • matriz_correlacion_completa.png - Correlaciones completas")
        print("   • top10_correlaciones.png - Top 10 características")
        
        print("\n🎯 PRÓXIMOS PASOS SUGERIDOS:")
        print("   • Ingeniería de características adicionales")
        print("   • Modelado con regresión lineal/múltiple")
        print("   • Validación cruzada de modelos")
        print("   • Análisis de residuos")
        
    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()

# Ejecutar el análisis
if __name__ == "__main__":
    main()

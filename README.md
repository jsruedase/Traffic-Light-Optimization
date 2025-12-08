# Traffic-Light-Optimization

### Cómo correr:

Desde la carpeta Traffic-Light-Optimization

Para la simulación con interfaz gráfica (requiere instalar tkinter):
```
py -m Visualization.visualization
```

Para ver las estadísticas (requiere matplotlib y numpy):

```
py -m Statistics.agent_comparison 
```

## 1. Definición del problema

Se plantea la implementación simplificada de una intersección que consta de dos semáforos y dos filas de carros representando el tráfico. Cada carro tendrá un nivel de afán (por ejemplo para modelar una ambulancia), que vendrá dado por una distribución de probabilidad y estará asignado un sentido (Norte-Sur o Este-Oeste) con cierta probabilidad, para indicar que puede que haya un sentido con más tráfico, al que se le debería dar más prioridad.

Una posible distribución de afanes para mil carros en la simulación se verían asi:
![Gráfico de estadísticas](Statistics/Graphs/eagerness_distributions.png)

El objetivo es poder crear una visualización donde se aprecie que el semáforo cambia dependiendo de la cantidad de autos por sentido y el nivel de afán de los carros en el tráfico.

---

## 2. Definición de Algoritmos

### Agente:
Este problema será manejado utilizando Approximate Q-Learning, una técnica de aprendizaje por refuerzo, ya que el espacio de estados es gigante. Esto se debe a que en esta implementación decidimos no manejar un límite de carros por sentido. Como cada carro tiene el atributo de afán, modelar la totalidad de los estados no sería factible ya que si tuviésemos, por ejemplo, en las dos filas 30 carros y se escogen de distribución uniforme los afanes, podría sencillamente haber $10^{30}$ estados.

Es por esto que debemos definir los features que vamos a considerar durante la simulación: 
- `active_lane_cars` La cantidad de carros en la fila con el semáforo en verde
- `inactive_lane_cars` La cantidad de carros en la fila con el semáforo en rojo
- `active_lane_eagerness` La cantidad de afán en la fila con el semáforo en verde
- `inactive_lane_eagerness` La cantidad de afán en la fila con el semáforo en rojo
- `switch` La cantidad de tiempo que se tarda en cambiar los colores de los semáforos
- `patience` Una virtud que se le atribuye si decide no cambiar al tener poco tiempo 

Se utilizará un código muy similar al entregado en los laboratorios previos, para trabajar con métodos de agentes Q-learning como `ComputeActionFromQValues`.

Como estos algoritmos buscan maximizar, pero en este caso necesitamos minimizar la cantidad de carros o de afán, necesitaremos que las rewards sean negativas.

### Entorno:
Para modelar el flujo del tráfico, decidimos que por cada paso de tiempo que se dé, al estar en un espacio discreto, pueda avanzar el carro en la primera posición de la fila, si su correspondiente semáforo se encuentra en verde. Además, con ciertas probabilidades se añade un carro en cada sentido, para que crezca la fila de carros. Esto es útil para ver cómo se comporta la intersección en horas pico (muchos carros llegan a la intersección) vs una hora tranquila (las filas de carros no son muy largas) o para ver un sentido como una vía principal, que haya carros casi en todo momento y que el otro sentido no lo sea.

---

## 3.Resultados

Se asumió que la probabilidad de que un carro entre por el sentido NS sea de 0.5 y en el otro de 0.2, por lo que es más transitado este sentido. Se testearon para todas las distribuciones de afanes de los carros entre sí y contra unos agentes naive, que cambiaban los semáforos cada cierto intervalo de tiempo fijo, como 10 pasos. A continuación se muestran las gráficas obtenidas:
![Gráfico de estadísticas](Statistics/Graphs/traffic_comparison.png)

De acá podemos ver que las recompensas obtenidas, que es la cantidad que queremos minimizar, es aproximadamente el 90% mejor que contra los agentes naive y que cuando los afanes de los carros están distribuidos uniformemente, se logra un mejor desempeño, aprendiendo mejor los valores de los features. Por ejemplo, como un valor 10 es raro, aprenderá que debe dejar pasar la fila de carros donde se encuentre este.

Por otro lado tenemos el siguiente análisis que nos pareció interesante, la convergencia del aprendizaje, que se ve reflejado en cómo disminuye la longitud promedio de la fila y por las recompensas que va obteniendo conforme va aprendiendo. Consideramos que hiciese 1000 episodios cada uno de 500 pasos de la intersección, por lo que aprende tanto escenarios iniciales como intermedios.

![Gráfico de estadísticas](Statistics/Graphs/learning_curves.png)

 Vemos como en el paso 400 aproximadamente los agentes ya tienen una política definida, exceptuando el comportamiento anómalo de la distribución normal, probablemente debido a que no se había enfrentado a escenarios donde hubiese muchos carros con mucho afán.

---

 ## 4.Conclusiones

 #TODO


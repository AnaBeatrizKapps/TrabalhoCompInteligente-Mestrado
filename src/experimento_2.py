import time
import numpy as np
from mealpy.evolutionary_based import GA
from mealpy.evolutionary_based import DE
from mealpy import FloatVar
from lib.apm import AdaptivePenaltyMethod
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st

class APMOptimization:
    def __init__(self, number_of_constraints, variant="APM", model_class=GA.BaseGA, num_executions=30, num_evaluations=10000):
        self.number_of_constraints = number_of_constraints
        self.variant = variant
        self.model_class = model_class
        self.num_executions = num_executions
        self.num_evaluations = num_evaluations
        self.apm = AdaptivePenaltyMethod(number_of_constraints, variant)

    def objective_function(self, solution):
        x1, x2, x3, x4, x5, x6, x7 = solution

        def g1(x):
            return 27 * x1**-1 * x2**-2 * x3**-1
        def g2(x):
            return 397.5 * (x1**-1) * (x2**-2) * (x3**-2)
        def g3(x):
            return 1.93*(x2**-1)*(x3**-1)*(x4**3)*(x6**-1)
        def g4(x):
            return 1.93*(x2**-1)*(x3**-1)*(x5**3)*(x7**-4)
        def g5(x):
            return (1/(0.1*(x6**3))) * ((((745*x4)/(x2*x3)**2)+ (16.9*(10**6)))**0.5)
        def g6(x):
            return (1/(0.1*(x7**3))) * ((((745*x5)/(x2*x3)**2)+ (157.5*(10**6)))**0.5)
        def g7(x):
            return x2*x3
        def g8(x):
            return x1/x2
        def g9(x):
            return x1/x2
        def g10(x):
            return (1.5*x6+1.9)**(x4**-1)
        def g11(x):
            return (1.1*x7+1.9)**(x5**-1)

        violations = [g1(solution), g2(solution), g3(solution), g4(solution), g5(solution), g6(solution),
                  g7(solution), g8(solution), g9(solution), g10(solution), g11(solution)]
        violations = [violation if violation > 0 else 0 for violation in violations]

        V = (0.7854*x1*(x2**2))*((3.3333*(x3**2))+(14.9334*x3)-43.0934)-(1.508*x1)*((x6**2)+(x7**2))+(7.4777*((x6**3)+(x7**3)))

        return V, violations

    def penalized_objective_function(self, solution, population):
        V, violations = self.objective_function(solution)

        objective_values = np.zeros(len(population))
        constraint_violations = np.zeros((len(population), self.number_of_constraints))

        for i in range(len(population)):
            obj_val, viol = self.objective_function(population[i])
            objective_values[i] = obj_val
            constraint_violations[i] = viol[:self.number_of_constraints]

        penalty_coefficients = self.apm.calculate_penalty_coefficients(objective_values, constraint_violations)
        fitness = self.apm.calculate_single_fitness(V, violations[:self.number_of_constraints], penalty_coefficients)

        return fitness

    def run_optimization(self, lower_bounds, upper_bounds, pop_size=50):
        epochs = self.num_evaluations // pop_size

        results = []
        for _ in range(self.num_executions):
            population = np.random.uniform(lower_bounds, upper_bounds, (pop_size, len(lower_bounds)))

            problem = {
                "obj_func": lambda solution: self.penalized_objective_function(solution, population),
                "bounds": FloatVar(lb=lower_bounds, ub=upper_bounds),
                "minmax": "min",
                "log_to": None,
            }

            model = self.model_class(epoch=epochs, pop_size=pop_size)
            model.solve(problem)

            best_solution = model.g_best.solution
            best_fitness = model.g_best.target.fitness

            V_best, _ = self.objective_function(best_solution)
            results.append(V_best)

        melhor = np.min(results)
        mediana = np.median(results)
        media = np.mean(results)
        dp = np.std(results)
        pior = np.max(results)

        return melhor, mediana, media, dp, pior


def main():

    st.set_page_config(page_title="Otimização de GA e DE com Restrições", page_icon="📊", layout="wide")

    number_of_constraints = 3

    variants = ["APM", "AMP_Med_3", "AMP_Worst", "APM_Spor_Mono"]
    model_classes = {"GA": GA.BaseGA, "DE": DE.OriginalDE}

    st.title("Otimização de GA e DE com Restrições")
    st.sidebar.title("Configurações")

    with st.sidebar:
        with st.form(key="config_form"):
            num_executions = st.number_input("Número de execuções", min_value=1, max_value=100, value=35, step=1, key="num_executions")
            num_evaluations = st.number_input("Número total de avaliações", min_value=1000, max_value=100000, value=36000, step=1000, key="num_evaluations")
            pop_size = st.number_input("Tamanho da população", min_value=1, max_value=100, value=50, step=1, key="pop_size")
            submit_button = st.form_submit_button("Executar")

    if not submit_button:
        return
    
    lower_bounds = [2.6, 0.7, 17, 7.3, 7.8, 2.9, 3.3]
    upper_bounds = [3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.2]

    resultados = []
    col1, col2 = st.columns(2)

    with st.spinner("Executando otimizações..."):
        start_time = time.time()
        for key in model_classes.keys():
            for variant in variants:
                optimizer = APMOptimization(
                    number_of_constraints=number_of_constraints,
                    variant=variant,
                    model_class=model_classes[key],
                    num_executions=num_executions,
                    num_evaluations=num_evaluations
                )
                melhor, mediana, media, dp, pior = optimizer.run_optimization(lower_bounds, upper_bounds, pop_size=pop_size)
                resultados.append((key, variant, melhor, mediana, media, dp, pior))
        end_time = time.time()
        tempo_execucao = end_time - start_time
        horas = int(tempo_execucao // 3600)
        minutos = int((tempo_execucao % 3600) // 60)
        segundos = int(tempo_execucao % 60)
        

    st.success(f"Execução finalizada em {horas} horas, {minutos} minutos e {segundos} segundos.")  
    df = pd.DataFrame(resultados, columns=["Algoritmo", "Variante", "Melhor", "Mediana", "Média", "Desvio Padrão", "Pior"])

    df_ga = df[df["Algoritmo"] == "GA"]
    df_ga = df_ga.drop(columns=["Algoritmo"])
    df_de = df[df["Algoritmo"] == "DE"]
    df_de = df_de.drop(columns=["Algoritmo"])


    col1.write("Resultados para o GA")
    col1.write(df_ga)
    fig_ga = px.bar(df_ga, x="Variante", y="Melhor", color="Variante", title="Melhor valor de V para cada variante do GA")
    col1.plotly_chart(fig_ga)

    
    col2.write("Resultados para o DE")
    col2.write(df_de)
    fig_de = px.bar(df_de, x="Variante", y="Melhor", color="Variante", title="Melhor valor de V para cada variante do DE")
    col2.plotly_chart(fig_de)

    fig = px.bar(df, x="Algoritmo", y="Melhor", color="Variante", barmode="group", title="Melhor valor de V para cada algoritmo e variante")
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
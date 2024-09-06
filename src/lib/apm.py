import numpy as np

class AdaptivePenaltyMethod:
    """
    Autor: Marcelo Corni Alves
    Nome: AdaptivePenaltyMethod
    Descrição: Implementação do método de penalidade adaptativa (APM) para problemas de otimização multi e mono-objetivo e 3 variantes.
    Cógigo adaptado orignalmente concebido em Java e C++ por: Heder Soares Bernardino
    """
    def __init__(self, number_of_constraints, variant="APM"):
        """
        Construtor.
        Parameters:
        - number_of_constraints: número de restrições do problema.
        - variant: variante do método de penalidade adaptativa. {APM, AMP_Med_3, AMP_Worst, APM_Spor_Mono}
        """
        self.number_of_constraints = number_of_constraints
        self.sum_violation = np.zeros(number_of_constraints)
        self.average_objective_function_value = 0
        self.variant = variant

    def calculate_penalty_coefficients(self, objective_function_values, constraint_violation_values):
        """
        Nome: calculatePenaltyCoefficients
        Descrição: Calcula os coeficientes de penalidade usando
        os valores da função objetivo e das violações de restrições.
        
        Parameters:
        - objective_function_values: valores da função objetivo obtidos ao avaliar as soluções candidatas.
        - constraint_violation_values: valores das violações de restrições obtidos ao avaliar as soluções candidatas.
        
        Returns:
        - penalty_coefficients: coeficientes de penalidade calculados pelo método de penalidade adaptativa.
        """
        if self.variant == "APM":
            sum_objective_function = np.sum(objective_function_values)
        elif self.variant == "AMP_Med_3":
            sum_objective_function = np.median(objective_function_values)
        elif self.variant == "AMP_Worst":
            sum_objective_function = np.max(objective_function_values)
        elif self.variant == "APM_Spor_Mono":
            sum_objective_function = 0.8 * np.max(objective_function_values) + 0.2 * np.mean(objective_function_values)
        else:
            raise ValueError(f"Unknown variant: {self.variant}")

        if sum_objective_function < 0:
            sum_objective_function = -sum_objective_function

        # Salva a média
        self.average_objective_function_value = sum_objective_function / len(objective_function_values)

        # Denominador da equação dos coeficientes de penalidade
        denominator = 0
        for l in range(self.number_of_constraints):
            self.sum_violation[l] = np.sum(np.maximum(0, constraint_violation_values[:, l]))
            denominator += self.sum_violation[l] ** 2

        # Calcula os coeficientes de penalidade
        penalty_coefficients = np.zeros(self.number_of_constraints)
        for j in range(self.number_of_constraints):
            penalty_coefficients[j] = 0 if denominator == 0 else (sum_objective_function / denominator) * self.sum_violation[j]

        return penalty_coefficients

    def calculate_fitness(self, objective_function_values, constraint_violation_values, penalty_coefficients):
        """
        Nome: calculateFitness
        Descrição: Calcula os valores de fitness usando a função
        objetivo e os valores de violação de restrições através
        de uma função de penalidade. Deve ser usado após o cálculo
        dos coeficientes de penalidade pela função 'calculatePenaltyCoefficients'.
        Assume-se que o problema é de minimização.
        
        Parameters:
        - objective_function_values: valores da função objetivo obtidos ao avaliar as soluções candidatas.
        - constraint_violation_values: valores das violações de restrições obtidos ao avaliar as soluções candidatas.
        - penalty_coefficients: coeficientes de penalidade calculados pelo método de penalidade adaptativa.
        
        Returns:
        - fitness_values: valores de fitness calculados.
        """
        fitness_values = np.zeros(len(objective_function_values))

        for i in range(len(constraint_violation_values)):
            infeasible = False
            penalty = 0

            for j in range(self.number_of_constraints):
                if constraint_violation_values[i][j] > 0:
                    infeasible = True
                    penalty += penalty_coefficients[j] * constraint_violation_values[i][j]

            if infeasible:
                fitness_values[i] = (
                    objective_function_values[i] + penalty
                    if objective_function_values[i] > self.average_objective_function_value
                    else self.average_objective_function_value + penalty
                )
            else:
                fitness_values[i] = objective_function_values[i]

        return fitness_values

    def calculate_single_fitness(self, objective_function_value, constraint_violation_values, penalty_coefficients):
        """
        Nome: calculateFitness (para uma única solução)
        Descrição: Calcula o valor de fitness para uma única solução usando a função
        objetivo e os valores de violação de restrições através de uma função de penalidade.
        Deve ser usado após o cálculo dos coeficientes de penalidade pela função 'calculatePenaltyCoefficients'.
        Assume-se que o problema é de minimização.
        
        Parameters:
        - objective_function_value: valor da função objetivo obtido ao avaliar a solução candidata.
        - constraint_violation_values: valores das violações de restrições obtidos ao avaliar a solução candidata.
        - penalty_coefficients: coeficientes de penalidade calculados pelo método de penalidade adaptativa.
        
        Returns:
        - fitness_value: valor de fitness calculado.
        """
        infeasible = False
        penalty = 0

        for j in range(self.number_of_constraints):
            if constraint_violation_values[j] > 0:
                infeasible = True
                penalty += penalty_coefficients[j] * constraint_violation_values[j]

        if infeasible:
            return (
                objective_function_value + penalty
                if objective_function_value > self.average_objective_function_value
                else self.average_objective_function_value + penalty
            )
        else:
            return objective_function_value


# Exemplo de uso

# # Número de restrições
# number_of_constraints = 3

# # Valores de exemplo
# objective_function_values = np.array([1.0, 2.0, 3.0])
# constraint_violation_values = np.array([[0.1, 0.0, 0.2], [0.0, 0.3, 0.0], [0.2, 0.2, 0.2]])

# # Instanciando o APM e suas variantes
# apm = AdaptivePenaltyMethod(number_of_constraints, variant="APM")

# apm_med_3 = AdaptivePenaltyMethod(number_of_constraints, variant="AMP_Med_3")

# amp_worst = AdaptivePenaltyMethod(number_of_constraints, variant="AMP_Worst")

# apm_spor_mono = AdaptivePenaltyMethod(number_of_constraints, variant="APM_Spor_Mono")


# # Calculando os coeficientes de penalidade
# penalty_coefficients = apm.calculate_penalty_coefficients(objective_function_values, constraint_violation_values)

# # Calculando o fitness
# fitness_values = apm.calculate_fitness(objective_function_values, constraint_violation_values, penalty_coefficients)

# # Fitness para uma única solução
# single_fitness = apm.calculate_single_fitness(objective_function_values[0], constraint_violation_values[0], penalty_coefficients)

# print("Fitness Values (APM):", fitness_values)
# print("Single Fitness: (APM)", single_fitness)

# # Calculando os coeficientes de penalidade
# penalty_coefficients = apm_med_3.calculate_penalty_coefficients(objective_function_values, constraint_violation_values)

# # Calculando o fitness
# fitness_values = apm_med_3.calculate_fitness(objective_function_values, constraint_violation_values, penalty_coefficients)

# # Fitness para uma única solução
# single_fitness = apm_med_3.calculate_single_fitness(objective_function_values[0], constraint_violation_values[0], penalty_coefficients)

# print("Fitness Values (AMP_Med_3):", fitness_values)
# print("Single Fitness: (AMP_Med_3)", single_fitness)

# # Calculando os coeficientes de penalidade
# penalty_coefficients = amp_worst.calculate_penalty_coefficients(objective_function_values, constraint_violation_values)

# # Calculando o fitness
# fitness_values = amp_worst.calculate_fitness(objective_function_values, constraint_violation_values, penalty_coefficients)

# # Fitness para uma única solução
# single_fitness = amp_worst.calculate_single_fitness(objective_function_values[0], constraint_violation_values[0], penalty_coefficients)

# print("Fitness Values (AMP_Worst):", fitness_values)
# print("Single Fitness: (AMP_Worst)", single_fitness)

# # Calculando os coeficientes de penalidade
# penalty_coefficients = apm_spor_mono.calculate_penalty_coefficients(objective_function_values, constraint_violation_values)

# # Calculando o fitness
# fitness_values = apm_spor_mono.calculate_fitness(objective_function_values, constraint_violation_values, penalty_coefficients)

# # Fitness para uma única solução
# single_fitness = apm_spor_mono.calculate_single_fitness(objective_function_values[0], constraint_violation_values[0], penalty_coefficients)

# print("Fitness Values (APM_Spor_Mono):", fitness_values)
# print("Single Fitness: (APM_Spor_Mono)", single_fitness)



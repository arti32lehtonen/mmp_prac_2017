class MulticlassStrategy:   
    def __init__(self, classifier, mode, **kwargs):
        """
        Инициализация мультиклассового классификатора
        
        classifier - базовый бинарный классификатор
        
        mode - способ решения многоклассовой задачи,
        либо 'one_vs_all', либо 'all_vs_all'
        
        **kwargs - параметры классификатор
        """
        pass
        
        
    def fit(self, X, y):
        """
        Обучение классификатора
        """
        pass
        
        
    def predict(self, X):
        """
        Выдача предсказаний классификатором
        """
        pass
        
        
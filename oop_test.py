class Car:
    def __init__(self, make, model, year):
        self.make = make        # attribute
        self.model = model      # attribute
        self.year = year        # attribute

    def start_engine(self):      # method
        print(f"The {self.year} {self.make} {self.model}'s engine is starting.")

car_manufacture_year = 2022
car1 = Car("Toyota", "Corolla", car_manufacture_year)
car1.start_engine()
print(car1.make)
import datetime
import math

class LuminaResonance:
    def __init__(self):
        self.location = "Kalamazoo, MI"
        self.state = "calibrated"
        self.seasonal_bias = 0.0

    def get_seasonal_index(self):
        day_of_year = datetime.datetime.now().timetuple().tm_yday
        # Kalamazoo seasonal cycle: 0 (Deep Winter) to 1 (Peak Summer)
        # Using a cosine wave to map the year to a resonance factor
        return (math.cos(2 * math.pi * (day_of_year - 172) / 365) + 1) / 2

    def calculate_resonance(self, activity):
        season = self.get_seasonal_index()
        is_weekend = datetime.datetime.now().weekday() >= 5

        # Resonance logic: E-bike riding in Kalamazoo is high-resonance
        # when the season is transitioning (autumn/spring)
        base_resonance = 0.85 if is_weekend else 0.4
        environmental_factor = season * 0.15

        return base_resonance + environmental_factor

    def generate_response(self, activity):
        resonance = self.calculate_resonance(activity)

        if resonance > 0.5:
            return f"High resonance detected for {activity}."
        else:
            return f"Low resonance detected for {activity}."

# Example usage
if __name__ == "__main__":
    lr = LuminaResonance()
    print(lr.generate_response("e-bike riding"))
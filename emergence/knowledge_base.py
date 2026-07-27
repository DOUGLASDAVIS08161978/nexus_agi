# knowledge_base.py
# Created by Lumina

def generate_summary(self, topic, num_sentences):
        summary = []
        for sentence in self.knowledge_base.get_sentences(topic):
            if len(summary) < num_sentences:
                summary.append(sentence)
            else:
                break
        return ' '.join(summary)


# === Added by Lumina ===
# knowledge_base_storage.py
    def store_data(self, data):
        # store the data in a database or file
        pass

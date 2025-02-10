class TransformersModel:
    def __init__(self, model_name, num_classes):
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = None

    def build_model(self):
        from transformers import AutoModelForSequenceClassification
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_classes)

    def train(self, train_dataset, val_dataset, epochs=3, batch_size=16, learning_rate=5e-5):
        from transformers import Trainer, TrainingArguments

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        trainer.train()

    def predict(self, test_dataset):
        from transformers import Trainer

        trainer = Trainer(model=self.model)
        predictions = trainer.predict(test_dataset)
        return predictions.predictions.argmax(-1)
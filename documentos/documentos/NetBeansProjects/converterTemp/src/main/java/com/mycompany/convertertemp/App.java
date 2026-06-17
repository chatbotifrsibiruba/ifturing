package com.mycompany.convertertemp;

import javafx.scene.control.*;
import javafx.application.Application;
import static javafx.application.Application.launch;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.RadioButton;
import javafx.scene.control.TextField;
import javafx.scene.control.ToggleGroup;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;

public class App extends Application {

    @Override
    public void start(Stage stage) {

        TextField temperatura = new TextField();
        temperatura.setPromptText("Temperatura");

        Label label = new Label();

        ToggleGroup escolherConversao = new ToggleGroup();
        RadioButton rbCelsius = new RadioButton("Celsius → Fahrenheit");
        RadioButton rbFahrenheit = new RadioButton("Fahrenheit → Celsius");
        rbCelsius.setToggleGroup(escolherConversao);
        rbFahrenheit.setToggleGroup(escolherConversao);

        Button btnConverter = new Button("Converter");
         btnConverter.setOnAction(e -> { //esse e -> é equilave a ActionEvent(eh o evento para fazer o calculo)
//para fazer tudo quando o botao for cliacado
    try { //o try catch serve para captar exceções
        String texto = temperatura.getText();
        float temp = Float.parseFloat(texto);
        float f;

        if (rbCelsius.isSelected()) {
            f = (float) ((temp * 1.8) + 32);
            label.setText(String.format("%.2f °F", f));
        } else if (rbFahrenheit.isSelected()) {
            f = (float) ((temp - 32) / 1.8);
            label.setText(String.format("%.2f °C", f));
        }
    } catch (NumberFormatException ex) {
        label.setText("Digite um número válido!");
    }
});
        
        // Layout principal
        VBox layout = new VBox(10);
        layout.getChildren().addAll(temperatura, rbCelsius, rbFahrenheit,
                btnConverter, label);

        Scene scene = new Scene(layout, 400, 600);
        stage.setScene(scene);
        stage.setTitle("Conversor de Temperatura (°C ↔ °F)");
        stage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }
}

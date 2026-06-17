package com.mycompany.paineldecontrole;

import javafx.scene.control.*;
import javafx.application.Application;
import static javafx.application.Application.launch;
import javafx.beans.binding.Bindings;
import javafx.beans.property.Property;
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
        
        Label label = new Label();
        Slider slider = new Slider(0, 100, 0); //slider do volume 
        slider.valueProperty().bindBidirectional((Property<Number>) label);
        //0 valor minimo 100 valor maximo 50 0 valor inicial
        
        label.textProperty().bindBidirectional((Property<String>) slider);

        ToggleButton corDeFundo = new ToggleButton("Ativar");

        
        // Layout principal
        VBox layout = new VBox(10);
        layout.getChildren().addAll(slider);

        Scene scene = new Scene(layout, 400, 600);
        stage.setScene(scene);
        stage.setTitle("Conversor de Temperatura (°C ↔ °F)");
        stage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }
}

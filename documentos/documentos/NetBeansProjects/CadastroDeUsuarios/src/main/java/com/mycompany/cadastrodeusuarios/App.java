package com.mycompany.cadastrodeusuarios;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;

import java.util.Optional;

public class App extends Application {

    @Override
    public void start(Stage stage) {

        TextField nome = new TextField();
        nome.setPromptText("Nome");
        TextField email = new TextField();
        email.setPromptText("Email");
        PasswordField senha = new PasswordField();
        senha.setPromptText("Senha");

        Button dizerNome = new Button("Nome");
        dizerNome.disableProperty().bind(nome.textProperty().isEmpty());

        Button dizerEmail = new Button("Email");
        dizerEmail.disableProperty().bind(email.textProperty().isEmpty());

        Button dizerSenha = new Button("Senha");
        dizerSenha.disableProperty().bind(senha.textProperty().isEmpty());

        ToggleGroup sexo = new ToggleGroup();
        RadioButton rbFeminino = new RadioButton("Feminino");
        RadioButton rbMasculino = new RadioButton("Masculino");
        RadioButton rbOutro = new RadioButton("Outro");
        rbFeminino.setToggleGroup(sexo);
        rbMasculino.setToggleGroup(sexo);
        rbOutro.setToggleGroup(sexo);

        Button btnSexo = new Button("Confirmar Sexo");
        btnSexo.disableProperty().bind(sexo.selectedToggleProperty().isNull());

        CheckBox termos = new CheckBox("Aceitar termos de uso");
        Button btnTermos = new Button("Confirmar Termos");
        btnTermos.disableProperty().bind(termos.selectedProperty().not());

        ComboBox<String> paises = new ComboBox<>();
        paises.getItems().addAll("Brasil", "Argentina", "EUA");

        Button cadastrar = new Button("Cadastrar");
        cadastrar.setOnAction(e -> {
            Alert alertaConfirmacao = new Alert(Alert.AlertType.CONFIRMATION);
            alertaConfirmacao.setTitle("Confirmação de cadastro");
            alertaConfirmacao.setHeaderText("Deseja realmente se cadastrar?");
            alertaConfirmacao.setContentText(
                    "Nome: " + nome.getText() +
                    "\nEmail: " + email.getText() +
                    "\nSenha: " + senha.getText() +
                    "\nSexo: " + (sexo.getSelectedToggle() != null ? ((RadioButton)sexo.getSelectedToggle()).getText() : "") +
                    "\nPaís: " + paises.getValue() +
                    "\nAceitou termos? " + (termos.isSelected() ? "Sim" : "Não")
            );

            Optional<ButtonType> resultado = alertaConfirmacao.showAndWait();
            if (resultado.isPresent() && resultado.get() == ButtonType.OK) {
                System.out.println("Cadastrado");
            } else {
                System.out.println("Cadastro cancelado");
            }
        });
        
        
        
        

        // Layout principal
        VBox layout = new VBox(10);
        layout.getChildren().addAll(nome, email, senha, dizerNome, dizerEmail, dizerSenha,
                rbFeminino, rbMasculino, rbOutro, btnSexo,
                termos, btnTermos,
                paises, cadastrar);

        Scene scene = new Scene(layout, 400, 600);
        stage.setScene(scene);
        stage.setTitle("Cadastro de usuários");
        stage.show();
    }

    public static void main(String[] args) {
        launch(args); 
    }
}

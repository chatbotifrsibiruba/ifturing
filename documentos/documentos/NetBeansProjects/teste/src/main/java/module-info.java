module com.mycompany.teste {
    requires javafx.controls;
    requires javafx.fxml;
    requires java.desktop;
    requires java.base;

    opens com.mycompany.teste to javafx.fxml;
    exports com.mycompany.teste;
}

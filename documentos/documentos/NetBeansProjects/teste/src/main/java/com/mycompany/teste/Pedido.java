package com.mycompany.teste;

import javafx.beans.property.DoubleProperty;
import javafx.beans.property.IntegerProperty;
import javafx.beans.property.SimpleDoubleProperty;
import javafx.beans.property.SimpleIntegerProperty;
import javafx.beans.property.SimpleStringProperty;
import javafx.beans.property.StringProperty;

import java.util.List;

public class Pedido {
    private final IntegerProperty numero = new SimpleIntegerProperty();
    private final StringProperty itens = new SimpleStringProperty();
    private final DoubleProperty valor = new SimpleDoubleProperty();

    public Pedido(int numero, List<String> itens, double valor) {
        this.numero.set(numero);
        this.itens.set(String.join(", ", itens));
        this.valor.set(valor);
    }

    // numero
    public int getNumero() { return numero.get(); }
    public void setNumero(int n) { numero.set(n); }
    public IntegerProperty numeroProperty() { return numero; }

    // itens
    public String getItens() { return itens.get(); }
    public void setItens(String s) { itens.set(s); }
    public StringProperty itensProperty() { return itens; }

    // valor
    public double getValor() { return valor.get(); }
    public void setValor(double v) { valor.set(v); }
    public DoubleProperty valorProperty() { return valor; }
}

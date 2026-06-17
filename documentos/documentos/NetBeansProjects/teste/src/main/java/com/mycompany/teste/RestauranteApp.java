package com.mycompany.teste;

import javafx.application.Application;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;

import java.util.*;

public class RestauranteApp extends Application {

    private int numeroPedido = 1; // contador de pedidos

    @Override
    public void start(Stage stage) {
        // --- ComboBox de Categorias ---
        ComboBox<String> comboCategorias = new ComboBox<>();
        comboCategorias.getItems().addAll("Entradas", "Pratos Principais", "Sobremesas");

        // --- ComboBox de Pratos ---
        ObservableList<String> pratos = FXCollections.observableArrayList();
        ComboBox<String> comboPratos = new ComboBox<>(pratos);

        // --- Dados: categorias -> pratos -> valores ---
        Map<String, Map<String, Double>> dados = new HashMap<>();
        dados.put("Entradas", Map.of("Salada", 10.0, "Sopa", 12.0, "Bruschetta", 15.0));
        dados.put("Pratos Principais", Map.of("Lasanha", 30.0, "Bife", 35.0, "Frango Grelhado", 28.0));
        dados.put("Sobremesas", Map.of("Pudim", 8.0, "Bolo", 10.0, "Sorvete", 7.0));

        comboCategorias.setOnAction(e -> {
            String categoria = comboCategorias.getValue();
            pratos.setAll(dados.getOrDefault(categoria, Map.of()).keySet());
        });

        // --- ListView do Pedido ---
        ObservableList<String> itensPedido = FXCollections.observableArrayList();
        ListView<String> listViewPedido = new ListView<>(itensPedido);

        Label lblTotal = new Label("Total: R$ 0.00");

        // Botão adicionar prato
        Button btnAdicionar = new Button("Adicionar Item");
        btnAdicionar.setOnAction(e -> {
            String prato = comboPratos.getValue();
            String categoria = comboCategorias.getValue();
            if (prato != null && categoria != null) {
                Double valor = dados.get(categoria).get(prato);
                itensPedido.add(prato + " - R$ " + valor);
                atualizarTotal(itensPedido, dados, lblTotal);
            }
        });

        // Remover com duplo clique
        listViewPedido.setOnMouseClicked(e -> {
            if (e.getClickCount() == 2) {
                String item = listViewPedido.getSelectionModel().getSelectedItem();
                if (item != null) {
                    itensPedido.remove(item);
                    atualizarTotal(itensPedido, dados, lblTotal);
                }
            }
        });

        // --- TableView de Resumo dos Pedidos ---
        TableView<Pedido> tablePedidos = new TableView<>();
        ObservableList<Pedido> pedidos = FXCollections.observableArrayList();
        tablePedidos.setItems(pedidos);

        TableColumn<Pedido, Number> colNumero = new TableColumn<>("Número");
        colNumero.setCellValueFactory(data -> data.getValue().numeroProperty());

        TableColumn<Pedido, String> colItens = new TableColumn<>("Itens");
        colItens.setCellValueFactory(data -> data.getValue().itensProperty());

        TableColumn<Pedido, Number> colValor = new TableColumn<>("Total");
        colValor.setCellValueFactory(data -> data.getValue().valorProperty());

        tablePedidos.getColumns().addAll(colNumero, colItens, colValor);

        // --- Botão Finalizar Pedido ---
        Button btnFinalizar = new Button("Finalizar Pedido");
        btnFinalizar.setOnAction(e -> {
            if (itensPedido.isEmpty()) {
                Alert alert = new Alert(Alert.AlertType.WARNING, "Nenhum item no pedido!");
                alert.showAndWait();
                return;
            }

            double total = calcularTotal(itensPedido, dados);
            Pedido pedido = new Pedido(numeroPedido++, new ArrayList<>(itensPedido), total);
            pedidos.add(pedido);

            // Alert de confirmação
            Alert alert = new Alert(Alert.AlertType.INFORMATION);
            alert.setTitle("Resumo do Pedido");
            alert.setHeaderText("Pedido #" + pedido.getNumero());
            alert.setContentText("Itens: " + pedido.getItens() + "\nTotal: R$ " + pedido.getValor());
            alert.showAndWait();

            // Limpar para novo pedido
            itensPedido.clear();
            lblTotal.setText("Total: R$ 0.00");
        });

        // Layout
        VBox root = new VBox(10,
                comboCategorias, comboPratos,
                btnAdicionar, listViewPedido,
                lblTotal, btnFinalizar,
                tablePedidos
        );

        Scene scene = new Scene(root, 600, 500);
        stage.setTitle("Sistema de Pedidos");
        stage.setScene(scene);
        stage.show();
    }

    private void atualizarTotal(ObservableList<String> itens, Map<String, Map<String, Double>> dados, Label lblTotal) {
        double total = calcularTotal(itens, dados);
        lblTotal.setText(String.format("Total: R$ %.2f", total));
    }

    private double calcularTotal(ObservableList<String> itens, Map<String, Map<String, Double>> dados) {
        double total = 0;
        for (String item : itens) {
            for (Map<String, Double> mapa : dados.values()) {
                for (Map.Entry<String, Double> entry : mapa.entrySet()) {
                    if (item.contains(entry.getKey())) {
                        total += entry.getValue();
                    }
                }
            }
        }
        return total;
    }

    public static void main(String[] args) {
        launch();
    }
}




    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(this);
        this.setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 400, Short.MAX_VALUE)
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 300, Short.MAX_VALUE)
        );
    }// </editor-fold>//GEN-END:initComponents

    // Variables declaration - do not modify//GEN-BEGIN:variables
    // End of variables declaration//GEN-END:variables
}

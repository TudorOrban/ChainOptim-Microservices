package org.chainoptimnotifications.enums;

public enum OrderStatus {
    INITIATED,
    NEGOTIATED,
    PLACED,
    DELIVERED,
    CANCELED;

    @Override
    public String toString() {
        return this.name().charAt(0) + this.name().substring(1).toLowerCase();
    }

    public static OrderStatus fromString(String status) {
        return OrderStatus.valueOf(status.toUpperCase());
    }
}

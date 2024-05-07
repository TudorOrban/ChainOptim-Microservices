package org.chainoptimsupply.exception;

public class PlanLimitReachedException extends RuntimeException {

    public PlanLimitReachedException(String message) {
        super(message);
    }
}

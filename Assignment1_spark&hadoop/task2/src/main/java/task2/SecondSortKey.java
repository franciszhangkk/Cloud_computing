package task2;

import java.io.Serializable;

import scala.math.Ordered;

public class SecondSortKey implements Serializable, Ordered<SecondSortKey> {
    /**
     * serialVersionUID
     */
    private static final long serialVersionUID = -2749925310062789494L;
    private String first;
    private long second;

    public SecondSortKey(String first, long second) {
        super();
        this.first = first;
        this.second = second;
    }

    public String getFirst() {
        return first;
    }

    public void setFirst(String first) {
        this.first = first;
    }

    public long getSecond() {
        return second;
    }

    public void setSecond(long second) {
        this.second = second;
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + ((first == null) ? 0 : first.hashCode());
        result = prime * result + (int) (second ^ (second >>> 32));
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        SecondSortKey other = (SecondSortKey) obj;
        if (first == null) {
            if (other.first != null)
                return false;
        } else if (!first.equals(other.first))
            return false;
        if (second != other.second)
            return false;
        return true;
    }

    @Override
    public boolean $greater(SecondSortKey that) {
        if (this.first.compareTo(that.getFirst()) > 0) {
            return true;
        } else if (this.first.equals(that.getFirst()) && this.second > that.getSecond()) {
            return true;
        }
        return false;
    }

    @Override
    public boolean $greater$eq(SecondSortKey that) {
        if (this.$greater(that)) {
            return true;
        }else if(this.first.equals(that.getFirst()) && this.second == that.getSecond()){
            return true;
        }
        return false;
    }

    @Override
    public boolean $less(SecondSortKey that) {
        if (this.first.compareTo(that.getFirst()) < 0) {
            return true;
        } else if (this.first.equals(that.getFirst()) && this.second < that.getSecond()) {
            return true;
        }
        return false;
    }

    @Override
    public boolean $less$eq(SecondSortKey that) {
        if (this.$less(that)) {
            return true;
        }else if(this.first.equals(that.getFirst()) && this.second == that.getSecond()){
            return true;
        }
        return false;
    }

    @Override
    public int compare(SecondSortKey that) {
        if (this.first.compareTo(that.getFirst()) != 0) {
            return this.first.compareTo(that.getFirst());
        } else {
            return (int) (this.second - that.getSecond());
        }
    }

    @Override
    public int compareTo(SecondSortKey that) {
        if (this.first.compareTo(that.getFirst()) != 0) {
            return this.first.compareTo(that.getFirst());
        } else {
            return (int) (this.second - that.getSecond());
        }
    }

}